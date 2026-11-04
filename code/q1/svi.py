"""SVI calculus and grid diagnostics only: no fitting, repair, data loading or IO."""

from dataclasses import dataclass

import numpy as np

from ._validation import finite_arrays, scalar, value


@dataclass(frozen=True)
class SVIParameters:
    a: float
    b: float
    rho: float
    m: float
    eta: float

    def __post_init__(self):
        for name in ["a", "b", "rho", "m", "eta"]:
            object.__setattr__(self, name, scalar(name, getattr(self, name)))
        if self.b < 0 or abs(self.rho) >= 1 or self.eta <= 0:
            raise ValueError("SVI requires b>=0, |rho|<1, eta>0")


def _coordinates(k, parameters):
    if not isinstance(parameters, SVIParameters):
        raise ValueError("parameters must be validated SVIParameters")
    grid = finite_arrays(k=k)[0]
    with np.errstate(over="ignore", invalid="ignore"):
        x = grid - parameters.m
        radius = np.hypot(x, parameters.eta)
    if not np.all(np.isfinite(radius)):
        raise ValueError("SVI coordinates overflowed; rescale the grid/parameters")
    return grid, x, radius


def svi_total_variance(k, parameters, *, require_positive=True):
    """a+b*(rho*(k-m)+hypot(k-m,eta)); positivity is checked on supplied points."""
    _, x, radius = _coordinates(k, parameters)
    with np.errstate(over="ignore", invalid="ignore"):
        w = parameters.a + parameters.b * (parameters.rho * x + radius)
    w = value(w)
    if require_positive and np.any(np.asarray(w) <= 0):
        raise ValueError("SVI total variance must be strictly positive on the evaluation grid")
    return w


def svi_global_minimum(parameters):
    """Analytical min over all real k; a itself need not be nonnegative."""
    _coordinates(0, parameters)
    with np.errstate(over="ignore", invalid="ignore"):
        return value(parameters.a + parameters.b * parameters.eta *
                     np.sqrt((1 - parameters.rho) * (1 + parameters.rho)))


def svi_first_derivative(k, parameters):
    _, x, radius = _coordinates(k, parameters)
    with np.errstate(over="ignore", invalid="ignore"):
        return value(parameters.b * (parameters.rho + x / radius))


def svi_second_derivative(k, parameters):
    _, _, radius = _coordinates(k, parameters)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return value(parameters.b * (parameters.eta / radius) ** 2 / radius)


def variance_positivity(k, parameters):
    """Return all raw values and nonpositive locations, not a repaired surface."""
    grid = finite_arrays(k=k)[0]
    if grid.size == 0:
        raise ValueError("Positivity verification requires a nonempty evaluation grid")
    w = np.asarray(svi_total_variance(grid, parameters, require_positive=False))
    mask = w <= 0
    minimum = svi_global_minimum(parameters)
    return {"grid": grid, "total_variance": w, "positive_on_grid": bool(np.all(w > 0)),
            "nonpositive_locations": grid[mask], "nonpositive_values": w[mask],
            "global_minimum": minimum, "positive_globally": minimum > 0}


def examination_butterfly(k, parameters):
    """EXAMINATION formula, deliberately WITHOUT w''/2; do not relabel as complete."""
    grid = finite_arrays(k=k)[0]
    w = np.asarray(svi_total_variance(grid, parameters))
    first = svi_first_derivative(grid, parameters)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        g = (1 - grid * first / (2 * w)) ** 2 - first ** 2 / 4 * (1 / w + 0.25)
    return value(g)


def complete_butterfly(k, parameters):
    """Separately labelled complete expression: examination g + w''/2."""
    return value(examination_butterfly(k, parameters) + svi_second_derivative(k, parameters) / 2)


@dataclass(frozen=True)
class GridDiagnostics:
    grid: np.ndarray
    values: np.ndarray
    violation_mask: np.ndarray
    tolerance: float

    @property
    def count(self):
        return int(np.count_nonzero(self.violation_mask))

    @property
    def locations(self):
        return self.grid[self.violation_mask]

    @property
    def magnitudes(self):
        return -self.values[self.violation_mask]


def _negative_diagnostic(grid, values, tolerance):
    grid, values = finite_arrays(grid=grid, values=values)
    tolerance = scalar("tolerance", tolerance)
    if grid.size == 0 or tolerance < 0:
        raise ValueError("Diagnostics require a nonempty grid and nonnegative tolerance")
    return GridDiagnostics(grid, values, values < -tolerance, tolerance)


def butterfly_diagnostics(k, parameters, *, tolerance=0.0):
    """Retain distinct formulas, full grids/values, violation masks and magnitudes."""
    return {
        "examination": _negative_diagnostic(k, examination_butterfly(k, parameters), tolerance),
        "complete": _negative_diagnostic(k, complete_butterfly(k, parameters), tolerance),
    }


def calendar_total_variance_difference(earlier_w, later_w, earlier_maturity, later_maturity):
    """w_later - w_earlier, at the SAME supplied log-moneyness locations.

    Maturities must be ordered, in years. This never remaps strike grids or drops
    negative differences. Zero total variance is allowed for explicit T=0 anchors.
    """
    early, late, t1, t2 = finite_arrays(earlier_w=earlier_w, later_w=later_w,
                                       earlier_maturity=earlier_maturity, later_maturity=later_maturity)
    if np.any(early < 0) or np.any(late < 0) or np.any(t1 < 0) or np.any(t2 <= t1):
        raise ValueError("Require nonnegative variances and 0<=earlier_maturity<later_maturity")
    if np.any((t1 == 0) & (early != 0)):
        raise ValueError("An explicit T=0 anchor must have zero total variance")
    return value(late - early)


def calendar_diagnostics(k, earlier_w, later_w, earlier_maturity, later_maturity, *, tolerance=0.0):
    difference = calendar_total_variance_difference(earlier_w, later_w, earlier_maturity, later_maturity)
    return _negative_diagnostic(k, difference, tolerance)


def _term_structure(maturities, total_variance):
    times = finite_arrays(maturities=maturities)[0]
    w = finite_arrays(total_variance=total_variance)[0]
    if times.ndim != 1 or times.size < 2 or np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("Maturities must be a nonnegative, strictly increasing 1D array of length >=2")
    if w.ndim == 0 or w.size == 0 or w.shape[0] != times.size or np.any(w < 0):
        raise ValueError("Variance must be nonnegative, with maturity along its first axis")
    if times[0] == 0 and np.any(w[0] != 0):
        raise ValueError("An explicit T=0 anchor must have zero total variance")
    return times, w


def interpolate_total_variance(maturities, total_variance, query_maturity):
    """Piecewise linear w, NOT sigma; no extrapolation or implicit T=0 anchor.

    total_variance shape is (n_expiries, *grid_shape). Return shape is
    query_maturity.shape + grid_shape, or float for scalar/scalar evaluation.
    Nonmonotone inputs are retained: inspect negative forward variance separately.
    """
    times, w = _term_structure(maturities, total_variance)
    query = finite_arrays(query_maturity=query_maturity)[0]
    if np.any(query < times[0]) or np.any(query > times[-1]):
        raise ValueError("Extrapolation outside supplied maturities is not authorized")
    interval = np.clip(np.searchsorted(times, query, side="right") - 1, 0, len(times) - 2)
    weight = (query - times[interval]) / (times[interval + 1] - times[interval])
    weight = weight.reshape(query.shape + (1,) * (w.ndim - 1))
    return value((1 - weight) * w[interval] + weight * w[interval + 1])


@dataclass(frozen=True)
class ForwardVarianceDiagnostics:
    interval_start: np.ndarray
    interval_end: np.ndarray
    forward_variance: np.ndarray
    violation_mask: np.ndarray
    tolerance: float

    @property
    def indices(self):
        """Each row is (maturity_interval_index, *grid_indices)."""
        return np.argwhere(self.violation_mask)

    @property
    def magnitudes(self):
        return -self.forward_variance[self.violation_mask]


def forward_variance_diagnostics(maturities, total_variance, *, tolerance=0.0):
    times, w = _term_structure(maturities, total_variance)
    tolerance = scalar("tolerance", tolerance)
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    spacing = np.diff(times).reshape((-1,) + (1,) * (w.ndim - 1))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        forward = np.asarray(value(np.diff(w, axis=0) / spacing))
    return ForwardVarianceDiagnostics(times[:-1], times[1:], forward, forward < -tolerance, tolerance)


def fixed_log_moneyness_grid(lower, upper, points):
    """Explicit reusable evaluation grid; no empirical strike support is assumed."""
    lower, upper = scalar("lower", lower), scalar("upper", upper)
    if lower >= upper or isinstance(points, bool) or not isinstance(points, (int, np.integer)) or points < 2:
        raise ValueError("Require lower<upper and integer points>=2")
    return np.linspace(lower, upper, points, dtype=np.float64)
