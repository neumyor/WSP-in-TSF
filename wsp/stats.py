"""Moving-block bootstrap intervals for window-level losses of dependent time series."""

import math

import numpy as np


def block_bootstrap_ci(values, block, reps=4000, level=0.95, seed=0):
    """Return the mean and its interval; the interval is None when fewer than two blocks fit in the sample."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    block = max(1, int(block))
    if n < 2 * block:
        return float(values.mean()), None
    n_blocks = math.ceil(n / block)
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for r in range(reps):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        means[r] = np.concatenate([values[s : s + block] for s in starts])[:n].mean()
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return float(values.mean()), [float(low), float(high)]


def paired_gain(base_losses, new_losses, block, reps=4000):
    base_losses = np.asarray(base_losses, dtype=float)
    new_losses = np.asarray(new_losses, dtype=float)
    if base_losses.shape != new_losses.shape:
        raise ValueError("paired losses must have the same shape")
    gain, ci = block_bootstrap_ci(base_losses - new_losses, block, reps=reps)
    return dict(gain=gain, ci=ci, relative=gain / float(base_losses.mean()))
