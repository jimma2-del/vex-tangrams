import itertools
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import jax
import jax.numpy as jnp

from tangrams.sim import Sim, State, Action
from tangrams.sim.polygon import polygon_out_of_bounds, get_polygon_edges
from tangrams.movement import MovementPlanner, Plan
from tangrams.tangram import Tangram, SYMMETRY, INTERCHANGEABLE

# Rearranges the pieces from whatever layout they're in into a target `Tangram`.
#
# Phases:
#   1. DISASSEMBLE: spread the current layout out to the walls, OUTERMOST ring first (`MovementPlanner.plan_spread`).
#   2. STAGE: move each piece, one at a time, to its staging spot: where `MovementPlanner.spread_targets` would put it
#       if the TARGET were spread out (so it has a clear straight line in to its slot), already at its target angle.
#       The middle of the board is empty during this phase, so there's room to turn.
#       Each route comes from a grid search with every other piece held still: drive to a spot where the piece can
#       turn freely, rotate there, then drive to the staging spot.
#       If every remaining staging spot is taken by a piece that hasn't moved yet, one of those pieces first moves
#       aside to the nearest spot that isn't anyone's staging spot.
#   3. ASSEMBLE: `MovementPlanner.plan_return` into the target, INNERMOST ring first.

@dataclass
class Phase:
    name: str
    start: int # index into `Result.states` where this phase begins
    piece: int | None = None # the piece moving, for single-piece phases

@dataclass
class Result:
    states: list[State]
    phases: list[Phase] = field(default_factory=list)
    spread_plan: Plan | None = None
    assemble_plan: Plan | None = None
    final_pos: np.ndarray | None = None # where the pieces end up (after matching identical pieces)
    final_ang: np.ndarray | None = None

class Rearranger:
    def __init__(
        self,
        sim: Sim,
        mover: MovementPlanner | None = None,
        gap: float = 1.03, # final layout = target scaled up by this, so touching pieces end up slightly apart
        res: float = 0.05, # grid spacing (units) for route search
        rot_stride: int = 4, # only every `rot_stride`-th grid point is considered as a place to rotate
        wall_margin: float = 0.02, # routes keep this far from the bounds
        clearance: float = 0.05, # routes keep this far from other pieces (covers small tracking errors)
        pos_tol: float = 0.01,
        ang_tol: float = 0.005,
        vel_tol: float = 0.01,
        kp: float = 2.0,
        max_ctrl_steps: int = 1000, # per waypoint
    ):
        self.sim = sim
        self.mover = mover or MovementPlanner(sim, piece_margin=2 * clearance) # room between staging spots
        self.clearance = clearance
        self.gap = gap
        self.res, self.rot_stride, self.wall_margin = res, rot_stride, wall_margin
        self.pos_tol, self.ang_tol, self.vel_tol, self.kp = pos_tol, ang_tol, vel_tol, kp
        self.max_ctrl_steps = max_ctrl_steps

        verts = np.asarray(sim.SHAPES.vertices)
        self.r_max = np.linalg.norm(verts, axis=-1).max(axis=-1) # farthest vertex from each piece's center

        lo, hi = np.asarray(sim.settings.bounds, dtype=float)
        self.grid_n = np.floor((hi - lo) / res).astype(int) + 1 # fixed grid shape, so the jitted check never retraces

        self._free = jax.jit(self._free_impl)
        self._step = jax.jit(sim.step)

    ### TOP LEVEL ###

    def run(self, state: State, target: Tangram) -> Result:
        res = Result(states=[state])

        # 1. disassemble, outer ring first
        res.phases.append(Phase("DISASSEMBLE", 0))
        res.spread_plan = self.mover.plan_spread(state)
        states, ok = self.mover.run(state, res.spread_plan)
        if not ok:
            raise RuntimeError("disassembly failed or timed out")
        res.states += states[1:]

        # 2. stage every piece at its target angle, at the spot it'll slide in from
        res.final_pos, res.final_ang, staging = self.assign(res.states[-1], target)
        unstaged, moved_aside = list(range(7)), set()
        while unstaged:
            state = res.states[-1]
            for i in unstaged:
                path = self.plan_path(state, i, staging[i], res.final_ang[i])
                if path is not None:
                    unstaged.remove(i)
                    res.phases.append(Phase("STAGE", len(res.states) - 1, piece=i))
                    break
            else: # every route is blocked: move a piece that's in the way aside
                for i in unstaged:
                    path = None if i in moved_aside else self.plan_aside(state, i, staging, res.final_ang, unstaged)
                    if path is not None:
                        moved_aside.add(i)
                        res.phases.append(Phase("MOVE ASIDE", len(res.states) - 1, piece=i))
                        break
                else:
                    raise RuntimeError(f"no route to a staging spot for any of pieces {unstaged}")
            res.states += self.follow(state, i, path)[1:]

        # 3. assemble, inner ring first
        res.phases.append(Phase("ASSEMBLE", len(res.states) - 1))
        res.assemble_plan = self.mover.plan_return(res.states[-1], res.final_pos)
        states, ok = self.mover.run(res.states[-1], res.assemble_plan)
        if not ok:
            raise RuntimeError("assembly failed or timed out")
        res.states += states[1:]
        return res

    def assign(self, state: State, target: Tangram) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Picks each piece's target angle (among its symmetric equivalents, the closest to its current angle),
            computes the staging spots, and matches identical pieces to slots (shortest total travel to staging).
        Returns (final pos, final ang, staging pos) per piece."""
        pos, ang = np.asarray(state.pos), np.asarray(state.ang)
        final_pos = target.pos * self.gap

        best, best_cost = None, np.inf
        for swaps in itertools.product((False, True), repeat=len(INTERCHANGEABLE)):
            perm = np.arange(7)
            for (a, b), swap in zip(INTERCHANGEABLE, swaps):
                if swap:
                    perm[[a, b]] = perm[[b, a]]
            cost = np.sum(np.linalg.norm(pos - final_pos[perm], axis=-1))
            if cost < best_cost:
                best, best_cost = perm, cost

        final_pos = final_pos[best]
        final_ang = ang + _wrap(target.ang[best] - ang, SYMMETRY)
        target_state = state.replace(pos=jnp.asarray(final_pos), ang=jnp.asarray(final_ang),
            vel=jnp.zeros_like(state.vel), ang_vel=jnp.zeros_like(state.ang_vel))
        self.mover.validate_targets(target_state, final_pos)
        return final_pos, final_ang, self.mover.spread_targets(target_state)

    ### ROUTE SEARCH ###

    def plan_path(self, state: State, i: int, goal: np.ndarray, goal_ang: float) -> list[tuple[np.ndarray, float]] | None:
        """Route for piece `i` from where it is to (`goal`, `goal_ang`) with every other piece held still, as
            waypoints (pos, ang); consecutive waypoints differ in position OR angle, never both.
        Returns None if there's no route."""
        start, start_ang = np.asarray(state.pos[i]), float(state.ang[i])
        pts = self._grid(goal) # anchored so the goal is exactly a grid point
        goal_cell = tuple(np.round((goal - pts[0, 0]) / self.res).astype(int))

        free_a = self._free_grid(state, i, pts, start_ang)
        free_b = free_a if np.isclose(start_ang, goal_ang) else self._free_grid(state, i, pts, goal_ang)
        if not free_b[goal_cell]:
            return None

        # the start usually isn't on the grid: enter it at the nearest free point with a clear straight line
        start_cell = None
        base = np.floor((start - pts[0, 0]) / self.res).astype(int)
        for d in sorted(itertools.product((0, 1), repeat=2), key=lambda d: np.linalg.norm(pts[tuple(base + d)] - start)):
            cell = tuple(base + d)
            if all(0 <= c < n for c, n in zip(cell, free_a.shape)) and free_a[cell] and self._segment_free(state, i, start, pts[cell], start_ang):
                start_cell = cell
                break
        if start_cell is None:
            return None

        dist_a, dist_b = _bfs(free_a, start_cell), _bfs(free_b, goal_cell)

        # places it can turn from start_ang to goal_ang without hitting anything
        can_rotate = np.zeros_like(free_a)
        s = self.rot_stride
        sub = pts[::s, ::s]
        rot_free = free_a[::s, ::s] & free_b[::s, ::s] & (dist_a[::s, ::s] >= 0) & (dist_b[::s, ::s] >= 0)
        step = self.res / self.r_max[i] # rotation step that moves any vertex at most `res`
        for a in np.linspace(start_ang, goal_ang, int(np.ceil(abs(goal_ang - start_ang) / step)) + 1)[1:-1]:
            if not rot_free.any():
                break
            rot_free &= self._free_grid(state, i, sub, a)
        can_rotate[::s, ::s] = rot_free
        if np.isclose(start_ang, goal_ang):
            can_rotate = free_a & (dist_a >= 0) & (dist_b >= 0)
        if not can_rotate.any():
            return None

        total = np.where(can_rotate, dist_a + dist_b, np.iinfo(np.int32).max)
        rot_cell = np.unravel_index(np.argmin(total), total.shape)

        to_rot = [start] + [pts[c] for c in _descend(dist_a, rot_cell)[::-1]]
        to_goal = [pts[c] for c in _descend(dist_b, rot_cell)]
        to_goal[-1] = goal

        path = [(p, start_ang) for p in self._shortcut(state, i, to_rot, start_ang)]
        if not np.isclose(start_ang, goal_ang):
            path.append((path[-1][0], goal_ang))
        path += [(p, goal_ang) for p in self._shortcut(state, i, to_goal, goal_ang)[1:]]
        return path

    def plan_aside(self, state: State, i: int, staging: np.ndarray, staging_ang: np.ndarray, unstaged: list[int]):
        """Route for piece `i` (without turning) to the nearest reachable spot that doesn't overlap the staging spot of
            any other piece in `unstaged`. Returns None if there's no such spot."""
        start, ang = np.asarray(state.pos[i]), float(state.ang[i])
        pts = self._grid(start)
        start_cell = tuple(np.round((start - pts[0, 0]) / self.res).astype(int))

        mask = np.isin(np.arange(7), [k for k in unstaged if k != i])
        clear = self._free_grid(state, i, pts, ang, ghosts=(staging, staging_ang, mask))
        dist = _bfs(self._free_grid(state, i, pts, ang), start_cell)
        cand = clear & (dist >= 0)
        if not cand.any():
            return None
        goal = pts[np.unravel_index(np.argmin(np.where(cand, dist, np.iinfo(np.int32).max)), dist.shape)]
        return self.plan_path(state, i, goal, ang)

    def _grid(self, anchor: np.ndarray) -> np.ndarray:
        """(nx, ny, 2) grid points spaced `res` apart covering the bounds, with `anchor` exactly on a point."""
        lo = np.asarray(self.sim.settings.bounds[0], dtype=float)
        first = anchor - np.floor((anchor - lo) / self.res) * self.res
        xs, ys = (first[d] + self.res * np.arange(self.grid_n[d]) for d in range(2))
        return np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)

    def _free_grid(self, state: State, i: int, pts: np.ndarray, ang: float, ghosts=None) -> np.ndarray:
        flat = jnp.asarray(pts.reshape(-1, 2))
        return np.asarray(self._free(state, i, flat, ang, *self._ghosts(state, ghosts))).reshape(pts.shape[:-1])

    def _ghosts(self, state: State, ghosts: tuple[np.ndarray, np.ndarray, np.ndarray] | None):
        """(pos, ang, mask) of extra obstacles, one slot per piece; masked-out slots are ignored."""
        if ghosts is None:
            return state.pos, state.ang, jnp.zeros(7, dtype=bool)
        return tuple(jnp.asarray(g) for g in ghosts)

    def _segment_free(self, state: State, i: int, a: np.ndarray, b: np.ndarray, ang: float) -> bool:
        n = int(np.ceil(np.linalg.norm(b - a) / (self.res / 4))) + 1
        n_pad = max(64, 1 << (n - 1).bit_length()) # pad to a power of 2 to limit retracing
        pts = np.linspace(a, b, n)
        pts = np.concatenate((pts, np.repeat(pts[-1:], n_pad - n, axis=0)))
        return bool(np.all(np.asarray(self._free(state, i, jnp.asarray(pts), ang, *self._ghosts(state, None)))))

    def _shortcut(self, state: State, i: int, pts: list[np.ndarray], ang: float) -> list[np.ndarray]:
        """Drops waypoints that a straight line can skip over."""
        out = [pts[0]]
        k = 0
        while k < len(pts) - 1:
            j = k + 1
            while j + 1 < len(pts) and self._segment_free(state, i, pts[k], pts[j + 1], ang):
                j += 1
            out.append(pts[j])
            k = j
        return out

    def _free_impl(self, state: State, i: jax.Array, points: jax.Array, ang: jax.Array,
                   ghost_pos: jax.Array, ghost_ang: jax.Array, ghost_mask: jax.Array) -> jax.Array:
        """For each point: piece `i` there at angle `ang` is in bounds (with margin) and clear of every other piece,
            and of the "ghost" copies of pieces at `ghost_pos`/`ghost_ang` where `ghost_mask` is set."""
        bounds = self.sim.settings.bounds + np.array(((self.wall_margin,), (-self.wall_margin,)))
        shape_i = jax.tree.map(lambda x: jnp.asarray(x)[i], self.sim.SHAPES) # `i` is traced, so index a jax array
        others = jnp.concatenate((jax.vmap(get_polygon_edges)(self.sim.SHAPES, state.ang, state.pos),
                                  jax.vmap(get_polygon_edges)(self.sim.SHAPES, ghost_ang, ghost_pos)))
        not_self = jnp.concatenate((jnp.arange(7) != i, ghost_mask))

        def free(p):
            edges = get_polygon_edges(shape_i, ang, p)
            colls = jax.vmap(_within, in_axes=(None, 0, None))(edges, others, self.clearance)
            return ~(jnp.any(colls & not_self) | polygon_out_of_bounds(p, ang, shape_i, bounds))
        return jax.vmap(free)(points)

    ### EXECUTION ###

    def follow(self, state: State, i: int, path: list[tuple[np.ndarray, float]]) -> list[State]:
        """Drives piece `i` through `path` (stopping at each waypoint) while every other piece stays put."""
        states = [state]
        for pos, ang in path[1:]:
            for _ in range(self.max_ctrl_steps):
                vel, ang_vel = np.zeros((7, 2)), np.zeros(7)
                vel[i] = self._approach(np.asarray(state.pos[i]), np.asarray(state.vel[i]), pos,
                    self.sim.settings.max_vel, self.sim.settings.max_accel)
                ang_vel[i] = self._approach(np.array((state.ang[i],)), np.array((state.ang_vel[i],)), np.array((ang,)),
                    self.sim.settings.max_ang_vel, self.sim.settings.max_ang_accel)[0]

                state, failed = self._step(state, Action(target_vel=jnp.asarray(vel), target_ang_vel=jnp.asarray(ang_vel)))
                states.append(state)
                if failed or self.sim.check_failed(state):
                    raise RuntimeError(f"piece {i} hit something while following its route")

                pos_err = np.linalg.norm(np.asarray(state.pos[i]) - pos)
                ang_err = abs(float(state.ang[i]) - ang)
                speed = max(np.linalg.norm(np.asarray(state.vel[i])), abs(float(state.ang_vel[i])))
                if pos_err < self.pos_tol and ang_err < self.ang_tol and speed < self.vel_tol:
                    break
            else:
                raise RuntimeError(f"piece {i} timed out reaching waypoint {pos}, {ang:.3f}")
        return states

    def _approach(self, x: np.ndarray, v: np.ndarray, target: np.ndarray, max_v: float, max_a: float) -> np.ndarray:
        """Velocity that drives `x` to `target` in a straight line and stops on it (same rule as `MovementPlanner`)."""
        delta = target - x
        dist = np.linalg.norm(delta)
        dir = delta / (dist + 1e-12)
        dist_after = max(dist - max(float(v @ dir), 0) * self.sim.settings.ctrl_dt, 0)
        return dir * min(max_v, np.sqrt(2 * 0.9 * max_a * dist_after), self.kp * dist)

NEIGHBORS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]

def _can_move(free: np.ndarray, c: tuple[int, int], d: tuple[int, int]) -> bool:
    """8-connected move from cell `c` by `d`; diagonal moves also need both side cells free (no corner cutting)."""
    n = (c[0] + d[0], c[1] + d[1])
    if not (0 <= n[0] < free.shape[0] and 0 <= n[1] < free.shape[1]) or not free[n]:
        return False
    return d[0] == 0 or d[1] == 0 or (free[c[0] + d[0], c[1]] and free[c[0], c[1] + d[1]])

def _bfs(free: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Steps from `start` to every cell (-1 if unreachable)."""
    dist = np.full(free.shape, -1, dtype=np.int32)
    dist[start] = 0
    queue = deque([start])
    while queue:
        c = queue.popleft()
        for d in NEIGHBORS:
            n = (c[0] + d[0], c[1] + d[1])
            if _can_move(free, c, d) and dist[n] < 0:
                dist[n] = dist[c] + 1
                queue.append(n)
    return dist

def _descend(dist: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    """Cells from `cell` down to the BFS start, following decreasing distance."""
    free = dist >= 0
    path = [tuple(int(x) for x in cell)]
    while dist[path[-1]] > 0:
        c = path[-1]
        path.append(next((c[0] + d[0], c[1] + d[1]) for d in NEIGHBORS
            if _can_move(free, c, d) and dist[c[0] + d[0], c[1] + d[1]] == dist[c] - 1))
    return path

def _within(edges1: jax.Array, edges2: jax.Array, margin: float) -> jax.Array:
    """True if two convex polygons overlap or are closer than `margin` along some edge normal
        (`convex_polygon_intersect` with unit-length axes and a margin)."""
    start1, end1 = edges1
    start2, end2 = edges2
    edges = jnp.concatenate((end1 - start1, end2 - start2))
    axes = jnp.stack((-edges[:, 1], edges[:, 0]), axis=-1)
    axes /= jnp.linalg.norm(axes, axis=-1, keepdims=True) + 1e-12 # padding edges have length 0 and never separate

    p1, p2 = start1 @ axes.T, start2 @ axes.T
    separated = (jnp.max(p1, axis=0) + margin <= jnp.min(p2, axis=0)) | (jnp.max(p2, axis=0) + margin <= jnp.min(p1, axis=0))
    return ~separated.any()

def _wrap(x: np.ndarray, period: np.ndarray) -> np.ndarray:
    """`x` shifted by a multiple of `period` into [-period/2, period/2)."""
    return (x + period / 2) % period - period / 2
