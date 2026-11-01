import numpy as np

from dataclasses import dataclass

from tangrams.sim import Sim

# A target tangram: where each of the 7 pieces should end up.
#
# The frontend describes a tangram as 7 polygons (one per piece, in any order). `Tangram.from_polygons` works out
#   which polygon is which piece (by shape) and the position + angle that puts the sim's piece exactly on it.

NAMES = ("small tri 1", "small tri 2", "medium tri", "large tri 1", "large tri 2", "square", "parallelogram")

# rotating a piece by this much maps it onto itself (the square every 90 degrees, the parallelogram every 180)
SYMMETRY = np.array((2*np.pi,) * 5 + (np.pi/2, np.pi))

# groups of identical pieces; any of them can fill any of their slots
INTERCHANGEABLE = ((0, 1), (3, 4))

TOTAL_AREA = 16 # the 7 pieces assemble into a square of side 4

@dataclass
class Tangram:
    pos: np.ndarray # target center of mass of each piece, shape (n_shapes=7, n_dims=2)
    ang: np.ndarray # target angle of each piece, shape (n_shapes=7)

    @classmethod
    def from_polygons(cls, polygons, fit_tol: float = 0.02) -> "Tangram":
        """`polygons`: 7 lists of vertices (one per piece, any order, either winding), with +y UP.
        Any scale works: the figure is rescaled to the sim's units (total area 16) and centered on (0, 0).
        Raises ValueError if the polygons aren't the 7 tangram pieces, or if the parallelogram is mirrored
            (the pieces can only rotate, not flip over)."""
        polys = [np.asarray(p, dtype=float) for p in polygons]
        if len(polys) != 7:
            raise ValueError(f"expected 7 polygons, got {len(polys)}")

        scale = np.sqrt(TOTAL_AREA / sum(abs(_signed_area(p)) for p in polys))
        polys = [p * scale for p in polys]

        pos, ang = np.zeros((7, 2)), np.zeros(7)
        unused = set(range(7))
        for n, poly in enumerate(polys):
            fits = [(_fit(_vertices(i), poly), i) for i in sorted(unused)]
            (err, a), i = min(fits, key=lambda f: f[0][0])
            if err > fit_tol:
                if _fit(_vertices(6), poly * (-1, 1))[0] <= fit_tol:
                    raise ValueError(f"polygon {n} is a mirrored parallelogram; flip the whole tangram left-to-right "
                        "(pieces can't be turned over)")
                raise ValueError(f"polygon {n} doesn't match any remaining piece (best: {NAMES[i]}, error {err:.3f})")
            pos[i], ang[i] = poly.mean(axis=0), a # vertex mean is the center of mass for triangles & parallelograms
            unused.remove(i)

        areas = np.array([abs(_signed_area(_vertices(i))) for i in range(7)])
        pos -= areas @ pos / areas.sum() # center the whole figure on (0, 0)
        return cls(pos=pos, ang=ang)

    def polygons(self) -> list[np.ndarray]:
        """World-space vertices of each piece."""
        return [_vertices(i) @ _rot(self.ang[i]).T + self.pos[i] for i in range(7)]

def _vertices(i: int) -> np.ndarray:
    return np.asarray(Sim.SHAPES.vertices[i][:Sim.SHAPES.n_vertices[i]])

def _rot(a: float) -> np.ndarray:
    return np.array(((np.cos(a), -np.sin(a)), (np.sin(a), np.cos(a))))

def _signed_area(v: np.ndarray) -> float:
    x, y = v[:, 0], v[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

def _ccw(v: np.ndarray) -> np.ndarray:
    return v if _signed_area(v) > 0 else v[::-1]

def _fit(shape: np.ndarray, poly: np.ndarray) -> tuple[float, float]:
    """Best rotation of `shape` (centered vertices) onto `poly`, over every vertex correspondence.
    Returns (max vertex error, angle)."""
    if len(shape) != len(poly):
        return np.inf, 0.0
    shape, poly = _ccw(shape), _ccw(poly - poly.mean(axis=0))

    best = (np.inf, 0.0)
    for k in range(len(poly)):
        target = np.roll(poly, k, axis=0)
        cross = np.sum(shape[:, 0] * target[:, 1] - shape[:, 1] * target[:, 0])
        a = np.arctan2(cross, np.sum(shape * target))
        err = np.max(np.linalg.norm(shape @ _rot(a).T - target, axis=-1))
        best = min(best, (err, float(a)))
    return best
