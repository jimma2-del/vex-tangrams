import numpy as np

import jax
import jax.numpy as jnp

from dataclasses import dataclass

from tangrams.sim import Sim, State, Action
from tangrams.sim.polygon import polygon_out_of_bounds, get_polygon_edges, convex_polygon_intersect

# Moves the pieces between the assembled square and a "spread out" layout.
#
# Ordering rules:
#   - spreading out: pieces are grouped into rings by distance from the center (0, 0);
#       the OUTERMOST ring moves first, then the next ring in, ...
#   - returning: the INNERMOST ring moves back first, then the next ring out, ...
#
# Collision handling (a "braking guard"):
#   Before each action, we roll the sim forward assuming we apply the action for one control step
#       and then brake every piece to a stop, calling `Sim.check_failed` at every physics step.
#   A piece is only allowed to move if that rollout stays collision-free and in bounds; otherwise it
#       brakes (yields). Since the previously chosen action already had a safe braking rollout,
#       "everyone brakes" is always a valid fallback, so the sim never enters a failed state.

@dataclass
class Plan:
    targets: np.ndarray # target positions, shape (n_shapes=7, n_dims=2)
    waves: list[list[int]] # groups of piece indices, in the order they should start moving
    start_pos: np.ndarray # positions when the plan was made, shape (n_shapes=7, n_dims=2)
    n_started: int = 0 # number of waves that have started moving

class MovementPlanner:
    def __init__(
        self,
        sim: Sim,
        spread: float = 2.0, # spread-out layout = assembled positions scaled up by this factor about the center
        wave_start: float = 0.5, # start the next wave once the current one is this fraction of the way there
        ring_tol: float = 0.05, # pieces whose distances from center differ by less than this share a ring
        pos_tol: float = 0.01, # a piece is "arrived" when it is this close to its target...
        vel_tol: float = 0.01, # ...and moving slower than this
        kp: float = 2.0, # proportional gain for the final approach, keeps pieces from overshooting
        max_out: tuple[int, ...] = tuple(range(7)), # pieces pushed as far out as they fit (default: all of them)
        max_out_step: float = 0.01, # step size (units) when searching how far out a `max_out` piece fits
        max_out_tilt_step: float = 5.0, # angle step (degrees) when tilting a blocked piece's direction
        wall_margin: float = 0.05, # gap (units) `max_out` pieces keep from the bounds
        piece_margin: float = 0.0, # gap (units) `max_out` pieces keep from the piece that stopped them
    ):
        self.sim = sim
        self.spread = spread
        self.wave_start = wave_start
        self.ring_tol = ring_tol
        self.pos_tol = pos_tol
        self.vel_tol = vel_tol
        self.kp = kp
        self.max_out = max_out
        self.max_out_step = max_out_step
        self.max_out_tilt_step = max_out_tilt_step
        self.wall_margin = wall_margin
        self.piece_margin = piece_margin

        settings = sim.settings
        self.n_sim_steps_per_ctrl = int(round(settings.ctrl_dt / settings.sim_dt))
        # sim steps needed to brake from max velocity to a stop, plus a little extra
        self.n_brake_steps = int(np.ceil(settings.max_vel / settings.max_accel / settings.sim_dt)) + 2

        self._rollout_safe = jax.jit(self._rollout_safe_impl)
        self._step = jax.jit(sim.step)
        self._blocked = jax.jit(self._blocked_impl, static_argnums=2) # piece index is static

    ### PLANNING ###

    def rings(self, positions: np.ndarray, outer_first: bool) -> list[list[int]]:
        """Groups pieces by distance from the center (0, 0)."""
        dists = np.linalg.norm(positions, axis=-1)
        order = np.argsort(-dists if outer_first else dists, kind="stable")

        rings = []
        for i in order:
            if rings and abs(dists[i] - dists[rings[-1][0]]) < self.ring_tol:
                rings[-1].append(int(i))
            else:
                rings.append([int(i)])
        return rings

    def spread_targets(self, state: State) -> np.ndarray:
        """Moves each piece straight outward, away from the center, by scaling the layout up.
        Scaling (rather than moving every piece the same distance) keeps the gaps between neighbors
            growing; equal distances let inner pieces catch up to outer ones (e.g. small tri 2 vs parallelogram).
        Pieces in `self.max_out` are then pushed out to the wall (outer pieces first, so inner ones fit around them).
            Each goes straight out from the center if that reaches the wall; if another piece is in the way,
            the direction is tilted by the smallest angle (in `max_out_tilt_step` increments) that reaches the wall.
            A piece's path must stay clear of the targets of pieces placed before it (outer pieces, which leave first)
                and of the assembled positions of pieces placed after it (inner pieces, which are still at home)."""
        home = np.asarray(state.pos)
        targets = home * self.spread

        max_out = [i for ring in self.rings(home, outer_first=True) for i in ring if i in self.max_out]
        layout = np.where(np.isin(np.arange(len(home)), max_out)[:, None], home, targets) # unplaced pieces at home
        for i in max_out:
            radial = np.arctan2(home[i, 1], home[i, 0]) if np.linalg.norm(home[i]) > 1e-6 else 0.0
            tilts = np.arange(0, 90 + 1e-6, self.max_out_tilt_step)
            tilts = np.concatenate([(t,) if t == 0 else (t, -t) for t in tilts])

            best, best_dist = targets[i], -1.0
            for tilt in tilts:
                ang = radial + np.deg2rad(tilt)
                target, reached_wall = self._walk_out(state, layout, i, np.array((np.cos(ang), np.sin(ang))))
                dist = np.linalg.norm(target - home[i])
                if reached_wall:
                    best = target
                    break
                if dist > best_dist: # keep the farthest spot in case no direction reaches the wall
                    best, best_dist = target, dist
            targets[i] = layout[i] = best

        return targets

    def _walk_out(self, state: State, layout: np.ndarray, i: int, dir: np.ndarray) -> tuple[np.ndarray, bool]:
        """Steps piece `i` from its assembled position along `dir` until it would overlap another piece in `layout`
            or go within `wall_margin` of the bounds. Returns its last valid spot and whether the wall stopped it.
        If another piece stopped it, it backs off by `piece_margin` (but never past its assembled position)."""
        home = np.asarray(state.pos)[i]
        last = home
        trial = layout.copy()
        back_off = int(np.ceil(self.piece_margin / self.max_out_step))

        for n in range(int(np.ceil(2 * np.max(np.abs(self.sim.settings.bounds)) / self.max_out_step))):
            trial[i] = home + n * self.max_out_step * dir
            out_of_bounds, collided = np.asarray(self._blocked(state, jnp.asarray(trial), i))
            if out_of_bounds:
                return last, True
            if collided:
                return home + max(n - 1 - back_off, 0) * self.max_out_step * dir, False
            last = trial[i].copy()
        return last, False

    def _blocked_impl(self, state: State, targets: jax.Array, i: int) -> jax.Array:
        """(piece `i` within `wall_margin` of the bounds, piece `i` overlaps another piece) for layout `targets`."""
        bounds = self.sim.settings.bounds + np.array(((self.wall_margin,), (-self.wall_margin,))) # shrink min & max
        shape_i = jax.tree.map(lambda x: x[i], self.sim.SHAPES)
        out_of_bounds = polygon_out_of_bounds(targets[i], state.ang[i], shape_i, bounds)

        shape_edges = jax.vmap(get_polygon_edges)(self.sim.SHAPES, state.ang, targets)
        colls = jax.vmap(convex_polygon_intersect, in_axes=(None, 0))(shape_edges[i], shape_edges)
        collided = jnp.any(colls & (jnp.arange(len(targets)) != i))
        return jnp.stack((out_of_bounds, collided))

    def plan_spread(self, state: State) -> Plan:
        """Plan for moving the (assembled) pieces outward, outer ring first."""
        home = np.asarray(state.pos)
        targets = self.spread_targets(state)
        self.validate_targets(state, targets)
        return Plan(targets=targets, waves=self.rings(home, outer_first=True), start_pos=home)

    def plan_return(self, state: State, home: np.ndarray) -> Plan:
        """Plan for moving the pieces back to `home` (e.g. `sim.reset().pos`), inner ring first."""
        home = np.asarray(home)
        self.validate_targets(state, home)
        return Plan(targets=home, waves=self.rings(home, outer_first=False), start_pos=np.asarray(state.pos))

    def validate_targets(self, state: State, targets: np.ndarray):
        target_state = state.replace(pos=jnp.asarray(targets), vel=jnp.zeros_like(state.vel))
        if self.sim.check_out_of_bounds(target_state):
            raise ValueError("target layout is out of bounds; try a smaller `spread` factor")
        if self.sim.check_collided(target_state):
            raise ValueError("target layout has overlapping pieces; try a larger `spread` factor")

    ### CONTROL ###

    def desired_vel(self, state: State, targets: np.ndarray) -> np.ndarray:
        """Velocity that drives each piece to its target and stops on it, within the sim's limits."""
        settings = self.sim.settings

        delta = targets - np.asarray(state.pos)
        dist = np.linalg.norm(delta, axis=-1)
        dirs = delta / (dist[:, None] + 1e-12)

        # each command is held for a whole control step, so plan the stop from where the piece will be after it
        speed_along = np.maximum(np.sum(np.asarray(state.vel) * dirs, axis=-1), 0)
        dist_after = np.maximum(dist - speed_along * settings.ctrl_dt, 0)

        # fastest speed from which we can still stop in `dist_after` (with 10% slack), capped by max_vel
        speed = np.minimum.reduce((
            np.full_like(dist, settings.max_vel),
            np.sqrt(2 * 0.9 * settings.max_accel * dist_after),
            self.kp * dist,
        ))
        return dirs * speed[:, None]

    def wave_progress(self, state: State, plan: Plan, wave: list[int]) -> float:
        """Fraction of the distance (0 to 1) covered by the slowest piece in `wave`.
        Using the slowest piece (not the average) means a piece that is held up can't let inner rings jump ahead."""
        total = np.linalg.norm(plan.targets[wave] - plan.start_pos[wave], axis=-1)
        left = np.linalg.norm(plan.targets[wave] - np.asarray(state.pos)[wave], axis=-1)
        return float(np.min(1 - left / np.maximum(total, 1e-6)))

    def arrived(self, state: State, plan: Plan) -> np.ndarray:
        pos_err = np.linalg.norm(plan.targets - np.asarray(state.pos), axis=-1)
        speed = np.linalg.norm(np.asarray(state.vel), axis=-1)
        return (pos_err < self.pos_tol) & (speed < self.vel_tol)

    def done(self, state: State, plan: Plan) -> bool:
        return plan.n_started == len(plan.waves) and bool(np.all(self.arrived(state, plan)))

    def action(self, state: State, plan: Plan) -> Action:
        """Picks the next action. Updates `plan.n_started` as waves begin moving."""
        # start the next wave once the current one has made enough progress
        if plan.n_started == 0:
            plan.n_started = 1
        while (plan.n_started < len(plan.waves)
               and self.wave_progress(state, plan, plan.waves[plan.n_started - 1]) >= self.wave_start):
            plan.n_started += 1

        desired = self.desired_vel(state, plan.targets)

        # braking guard: add pieces one at a time in wave order, keeping only ones with a safe rollout
        active = [i for wave in plan.waves[:plan.n_started] for i in wave]
        moving = np.zeros(len(desired), dtype=bool)
        moving[active] = True

        if not self._safe(state, desired, moving): # common case: everything active can move
            moving[:] = False
            for i in active:
                moving[i] = True
                if not self._safe(state, desired, moving):
                    moving[i] = False # piece i yields (brakes) this step

        target_vel = np.where(moving[:, None], desired, 0.0)
        return Action(target_vel=jnp.asarray(target_vel), target_ang_vel=jnp.zeros(len(desired)))

    def _safe(self, state: State, desired: np.ndarray, moving: np.ndarray) -> bool:
        target_vel = np.where(moving[:, None], desired, 0.0)
        return bool(self._rollout_safe(state, jnp.asarray(target_vel)))

    def _rollout_safe_impl(self, state: State, target_vel: jax.Array) -> jax.Array:
        """True if applying `target_vel` for one control step, then braking to a stop, never fails."""
        go = Action(target_vel=target_vel, target_ang_vel=jnp.zeros(target_vel.shape[0]))
        brake = Action(target_vel=jnp.zeros_like(target_vel), target_ang_vel=jnp.zeros(target_vel.shape[0]))

        def scan_func(carry, i):
            state, failed = carry
            action = jax.tree.map(lambda g, b: jnp.where(i < self.n_sim_steps_per_ctrl, g, b), go, brake)
            state = self.sim.sim_step(state, action)
            return (state, failed | self.sim.check_failed(state)), None

        n_steps = self.n_sim_steps_per_ctrl + self.n_brake_steps
        (_, failed), _ = jax.lax.scan(scan_func, (state, jnp.array(False)), jnp.arange(n_steps))
        return ~failed

    ### RUNNING ###

    def run(self, state: State, plan: Plan, max_steps: int = 2000) -> tuple[list[State], bool]:
        """Executes `plan` in the sim. Returns the state after every control step and whether it finished.
        Stops early (returns False) if the sim reports a failure or `max_steps` runs out."""
        states = [state]
        for _ in range(max_steps):
            if self.done(state, plan):
                return states, True

            state, failed = self._step(state, self.action(state, plan))
            states.append(state)
            if failed or self.sim.check_failed(state):
                return states, False

        return states, False
