import statistics

import numpy as np


def random_inputs(specs) -> dict:
    """Deterministic inputs: floats are uniform in [0, 1), masks are 1, other integers (ids, positions,
    beam indices) are 0 so every index is in range."""
    rng = np.random.default_rng(0)
    feeds = {}
    for name, shape, dtype in specs:
        dtype = np.dtype(dtype)
        if dtype.kind == "f":
            feeds[name] = rng.random(shape, dtype=np.float32).astype(dtype)
        else:
            feeds[name] = (np.ones if "mask" in name else np.zeros)(shape, dtype=dtype)
    return feeds


def stats(v: list[float]) -> dict:
    return {"median": statistics.median(v), "min": min(v), "max": max(v), "mean": statistics.fmean(v)}
