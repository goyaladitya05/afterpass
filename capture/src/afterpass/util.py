import statistics

import numpy as np


def random_inputs(specs) -> dict:
    """Deterministic inputs: floats are uniform in [0, 1), integers and bools are 1."""
    rng = np.random.default_rng(0)
    feeds = {}
    for name, shape, dtype in specs:
        dtype = np.dtype(dtype)
        if dtype.kind == "f":
            feeds[name] = rng.random(shape, dtype=np.float32).astype(dtype)
        else:
            feeds[name] = np.ones(shape, dtype=dtype)
    return feeds


def stats(v: list[float]) -> dict:
    return {"median": statistics.median(v), "min": min(v), "max": max(v), "mean": statistics.fmean(v)}
