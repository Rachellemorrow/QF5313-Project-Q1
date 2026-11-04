"""Small, data-independent float64 validation helpers for Q1 mathematics."""

import numpy as np


def finite_arrays(**values):
    arrays = []
    for name, value in values.items():
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real, not complex")
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be real numeric input") from error
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        arrays.append(array)
    try:
        return np.broadcast_arrays(*arrays)
    except ValueError as error:
        raise ValueError("Input shapes are not broadcast-compatible") from error


def scalar(name, value):
    array = finite_arrays(**{name: value})[0]
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    return float(array)


def value(array):
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("Numerical result is not representable in float64; rescale inputs")
    return float(array) if array.ndim == 0 else array
