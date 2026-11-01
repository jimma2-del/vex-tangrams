import sys
import time

import numpy as np

from tangrams.sim import Sim, Settings
from tangrams.tangram import Tangram, NAMES, SYMMETRY
from tangrams.rearrange import Rearranger
from test_movement import check_order, check_center_order, show

# Rearranges the pieces through a chain of target tangrams (each one starts from where the previous one ended):
#   sim start square -> square from the reference picture -> big triangle -> big triangle turned 30 degrees
#       -> picture square turned 45 degrees -> back to the sim start square
# and checks that:
#   - the sim never fails (no collisions, nothing out of bounds)
#   - disassembly starts with the outermost pieces (center piece moves out last)
#   - every piece turns to its target angle while spread out (STAGE)
#   - assembly places the innermost target pieces first (center piece moves in first)
#   - every piece ends at its target position and angle
# Then plays the motion in a pygame window (pass --no-show to skip it; N / P jump to the next / previous target).

# Targets as the frontend would send them: 7 polygons, any order and scale, +y up.

# The reference picture (4x4 square, origin at its bottom-left corner), MIRRORED left-to-right:
#   the picture's parallelogram is the mirror image of the sim's, and pieces can't flip over.
PICTURE_SQUARE = [
    [(0, 4), (4, 4), (2, 2)], # large tri (top)
    [(4, 4), (4, 0), (2, 2)], # large tri (right)
    [(0, 2), (0, 0), (2, 0)], # medium tri (bottom-left)
    [(2, 2), (3, 1), (2, 0), (1, 1)], # square
    [(3, 1), (4, 0), (2, 0)], # small tri (bottom-right)
    [(1, 3), (2, 2), (1, 1)], # small tri (middle)
    [(0, 4), (1, 3), (1, 1), (0, 2)], # parallelogram (left)
]

# hypotenuse of 8 along the bottom, apex on top (also mirrored to get the right parallelogram)
BIG_TRIANGLE = [
    [(-4, 0), (0, 0), (-2, 2)], # large tri
    [(0, 0), (4, 0), (2, 2)], # large tri
    [(-2, 2), (0, 0), (0, 2)], # medium tri
    [(0, 2), (1, 1), (2, 2), (1, 3)], # square
    [(0, 0), (1, 1), (0, 2)], # small tri
    [(1, 3), (0, 4), (-1, 3)], # small tri
    [(0, 2), (1, 3), (-1, 3), (-2, 2)], # parallelogram
]

def mirror(polys):
    return [[(-x, y) for x, y in p] for p in polys]

def rotate(polys, deg):
    a = np.deg2rad(deg)
    r = np.array(((np.cos(a), -np.sin(a)), (np.sin(a), np.cos(a))))
    return [np.asarray(p, dtype=float) @ r.T for p in polys]

def run_case(rearranger, name, state, target):
    print(f"\n=== {name} ===")
    t0 = time.time()
    res = rearranger.run(state, target)
    env = rearranger.sim

    n_spread = res.phases[1].start + 1
    check_order("DISASSEMBLE", res.spread_plan, res.states[:n_spread])
    check_center_order("DISASSEMBLE", np.asarray(state.pos), res.states[:n_spread], outward=True)
    print("  STAGE: " + "  ->  ".join(f"{NAMES[p.piece]}{' (aside)' if p.name == 'MOVE ASIDE' else ''}@{p.start}"
        for p in res.phases if p.piece is not None))
    assembly = res.states[res.phases[-1].start:]
    check_order("ASSEMBLE", res.assemble_plan, assembly)
    check_center_order("ASSEMBLE", res.final_pos, assembly, outward=False)

    assert not any(bool(env.check_failed(s)) for s in res.states), f"{name}: check_failed triggered"

    final = res.states[-1]
    pos_err = np.max(np.linalg.norm(np.asarray(final.pos) - res.final_pos, axis=-1))
    ang_diff = np.asarray(final.ang) - res.final_ang
    ang_err = np.max(np.abs((ang_diff + SYMMETRY / 2) % SYMMETRY - SYMMETRY / 2))
    assert pos_err < 0.02 and ang_err < 0.01, f"{name}: final error pos {pos_err:.4f}, ang {ang_err:.4f}"
    print(f"  ok: {len(res.states) * env.settings.ctrl_dt:.1f}s sim time ({time.time() - t0:.1f}s to plan + run), "
          f"max final error pos {pos_err:.4f}, ang {np.rad2deg(ang_err):.2f} deg")
    return res

def main():
    env = Sim(settings=Settings(ctrl_dt=0.1, sim_dt=0.025))
    rearranger = Rearranger(env)
    start = env.reset()

    targets = [
        ("picture square", Tangram.from_polygons(mirror(PICTURE_SQUARE))),
        ("big triangle", Tangram.from_polygons(mirror(BIG_TRIANGLE))),
        ("big triangle, turned 30 deg", Tangram.from_polygons(rotate(mirror(BIG_TRIANGLE), 30))),
        ("picture square, turned 45 deg", Tangram.from_polygons(rotate(mirror(PICTURE_SQUARE), 45))),
        ("sim start square", Tangram(pos=np.asarray(start.pos) / rearranger.gap, ang=np.asarray(start.ang))),
    ]

    results, state = [], start
    for name, target in targets:
        res = run_case(rearranger, name, state, target)
        phases = [(p.name if p.piece is None else f"{p.name} {NAMES[p.piece]}", p.start) for p in res.phases]
        results.append((name, res.states, phases))
        state = res.states[-1]
    print(f"\nALL {len(results)} TARGETS PASSED")

    if "--no-show" not in sys.argv:
        show(env, results)

if __name__ == "__main__":
    main()
