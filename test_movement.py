import sys

import jax
import jax.numpy as jnp
import numpy as np

from tangrams.sim import Sim, Settings, RenderSettings
from tangrams.movement import MovementPlanner

# Runs the spread-out plan, then the return plan, on several starting layouts and checks that:
#   - the sim never fails (no collisions, nothing out of bounds)
#   - rings start moving in the right order (outer first going out, inner first coming back)
#   - independently of the planner, pieces start in order of distance from the center (center piece last out, first in)
#   - pieces only move outward while spreading and only inward while returning
#   - every piece ends up back where it started
# Then plays the motion in a pygame window (pass --no-show to skip it; N / P jump to the next / previous layout).

NAMES = ["small tri 1", "small tri 2", "medium tri", "large tri 1", "large tri 2", "square", "parallelogram"]

def packed_layout(env, center, seed, gap=0.05, step=0.02, n_dirs=12):
    """Puts piece `center` at (0, 0), then adds the other pieces one at a time (random order, direction and
        rotation), sliding it out from the center along `n_dirs` directions and keeping the closest fit (plus `gap`)."""
    rng = np.random.default_rng(seed)
    state = env.reset()
    check_collided = jax.jit(env.check_collided)
    ang = rng.uniform(0, 2*np.pi, 7)
    pos = np.array([(100.0 + 10*i, 100.0) for i in range(7)]) # park unplaced pieces far away
    pos[center] = 0

    for i in rng.permutation([i for i in range(7) if i != center]):
        best = None
        for dir_ang in rng.uniform(0, 2*np.pi) + np.linspace(0, 2*np.pi, n_dirs, endpoint=False):
            dir = np.array((np.cos(dir_ang), np.sin(dir_ang)))
            collided = lambda r: check_collided(
                state.replace(pos=jnp.asarray(_with(pos, i, r * dir)), ang=jnp.asarray(ang)))
            r = 0.0
            while collided(r):
                r += step
            r += gap
            while collided(r): # the extra gap can push it into another piece
                r += step
            if best is None or r < np.linalg.norm(best):
                best = r * dir
        pos[i] = best # tightest fit, to keep the layout compact

    state = state.replace(pos=jnp.asarray(pos), ang=jnp.asarray(ang))
    assert not env.check_failed(state), "generated layout is invalid"
    return state

def _with(pos, i, p):
    pos = pos.copy()
    pos[i] = p
    return pos

def start_steps(states, moved_tol=0.02):
    """Control step at which each piece first moves `moved_tol` away from where it started."""
    pos = np.stack([np.asarray(s.pos) for s in states])
    moved = np.linalg.norm(pos - pos[0], axis=-1)
    return np.argmax(moved > moved_tol, axis=0)

def check_order(phase, plan, states):
    starts = start_steps(states)
    print(f"  {phase}: " + "  ->  ".join(", ".join(f"{NAMES[i]}@{starts[i]}" for i in wave) for wave in plan.waves))

    for earlier, later in zip(plan.waves, plan.waves[1:]):
        assert max(starts[earlier]) <= min(starts[later]), \
            f"{phase}: {[NAMES[i] for i in later]} moved before {[NAMES[i] for i in earlier]}"

def check_center_order(phase, home, states, outward, ring_tol=0.05):
    """Independent of the planner's rings: ranks pieces by distance of their assembled position from the center.
    Spreading, a farther piece never starts after a closer one and the center piece starts last;
        returning, a closer piece never starts after a farther one and the center piece starts first."""
    starts = start_steps(states)
    dists = np.linalg.norm(home, axis=-1)
    for i in range(7):
        for j in range(7):
            if dists[i] > dists[j] + ring_tol: # i is clearly farther out than j
                first, second = (i, j) if outward else (j, i)
                assert starts[first] <= starts[second], \
                    f"{phase}: {NAMES[second]}@{starts[second]} started before {NAMES[first]}@{starts[first]}"

    center = int(np.argmin(dists))
    expected = starts.max() if outward else starts.min()
    assert starts[center] == expected, f"{phase}: center piece {NAMES[center]} didn't move {'last' if outward else 'first'}"

def check_direction(phase, states, outward, tol=1e-3):
    """Distance from the center never shrinks while spreading / never grows while returning."""
    dists = np.stack([np.linalg.norm(np.asarray(s.pos), axis=-1) for s in states])
    change = np.diff(dists, axis=0) * (1 if outward else -1)
    bad = np.where(change.min(axis=0) < -tol)[0]
    assert len(bad) == 0, f"{phase}: {[NAMES[i] for i in bad]} moved {'inward' if outward else 'outward'}"

def run_case(env, planner, name, state):
    print(f"\n=== {name} ===")
    home = np.asarray(state.pos)

    spread_plan = planner.plan_spread(state)
    spread_states, ok = planner.run(state, spread_plan)
    assert ok, f"{name}: spread plan failed or timed out"
    check_order("SPREAD", spread_plan, spread_states)
    check_center_order("SPREAD", home, spread_states, outward=True)
    check_direction("SPREAD", spread_states, outward=True)

    return_plan = planner.plan_return(spread_states[-1], home)
    return_states, ok = planner.run(spread_states[-1], return_plan)
    assert ok, f"{name}: return plan failed or timed out"
    check_order("RETURN", return_plan, return_states)
    check_center_order("RETURN", home, return_states, outward=False)
    check_direction("RETURN", return_states, outward=False)

    all_states = spread_states + return_states[1:]
    assert not any(bool(env.check_failed(s)) for s in all_states), f"{name}: check_failed triggered"

    final_err = np.max(np.linalg.norm(np.asarray(all_states[-1].pos) - home, axis=-1))
    assert final_err < 0.02, f"{name}: pieces didn't make it back (error {final_err:.3f})"
    print(f"  ok: {len(all_states) * env.settings.ctrl_dt:.1f}s, max final position error {final_err:.4f}")
    return all_states, [("SPREAD", 0), ("RETURN", len(spread_states) - 1)]

def main():
    env = Sim(settings=Settings(ctrl_dt=0.1, sim_dt=0.025))
    planner = MovementPlanner(env)

    cases = [("default square", env.reset())]
    cases += [(f"square in middle (seed {seed})", packed_layout(env, center=5, seed=seed)) for seed in range(3)]
    cases += [(f"{NAMES[i]} in middle", packed_layout(env, center=i, seed=10 + i)) for i in range(7) if i != 5]

    results = [(name, *run_case(env, planner, name, state)) for name, state in cases]
    print(f"\nALL {len(results)} LAYOUTS PASSED")

    if "--no-show" not in sys.argv:
        show(env, results)

def show(env, results):
    """Loops through every layout's motion until the window is closed. `results`: (name, states, phases) where
        phases is a list of (label, index of its first state).
    SPACE pauses, LEFT/RIGHT step while paused, N/P jump to the next/previous layout."""
    import pygame

    # render every frame up front (bigger scale than the sim default so it's easy to see)
    big_env = Sim(settings=env.settings, render_settings=RenderSettings(scale=64))
    render = jax.jit(big_env.render)
    print(f"\nRendering {sum(len(r[1]) for r in results)} frames...")
    frames = [[np.asarray(render(s))[:, ::-1, :] for s in states] for _, states, _ in results] # flip y: pygame +y is down

    pygame.init()
    pygame.display.set_caption("Tangrams Movement")
    screen = pygame.display.set_mode(big_env.render_size)
    font = pygame.font.Font(None, 32)
    clock = pygame.time.Clock()
    print("Playing (close the window to quit; SPACE pauses, LEFT/RIGHT step, N/P next/previous layout)")

    case, i, paused = 0, 0, False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: paused = not paused
                if event.key == pygame.K_RIGHT: i = min(i + 1, len(frames[case]) - 1)
                if event.key == pygame.K_LEFT: i = max(i - 1, 0)
                if event.key in (pygame.K_n, pygame.K_p):
                    case = (case + (1 if event.key == pygame.K_n else -1)) % len(frames)
                    i = 0

        name, _, phases = results[case] # phases: (label, first frame) in order
        screen.blit(pygame.surfarray.make_surface(frames[case][i]), (0, 0))
        phase = [label for label, start in phases if start <= i][-1]
        lines = [f"{case + 1}/{len(results)}: {name}",
                 f"{phase}   t = {i * env.settings.ctrl_dt:.1f}s" + ("   (paused)" if paused else "")]
        for row, line in enumerate(lines):
            screen.blit(font.render(line, True, (0, 0, 0)), (10, 10 + 28*row))
        pygame.display.flip()

        if not paused:
            i += 1
            if i == len(frames[case]): # hold the final frame for a second, then go to the next layout
                pygame.time.wait(1000)
                case, i = (case + 1) % len(frames), 0
        clock.tick(round(1 / env.settings.ctrl_dt)) # real time

if __name__ == "__main__":
    main()
