"""Exported window tensors and the time-ordered fit/check split of the validation windows."""

import os

import numpy as np
import torch


def load_windows(base_dir, name):
    path = os.path.join(base_dir, f"{name}_data.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing window file: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data["inputs"].float(), data["targets"].float()


def split_validation(n_windows, lookback, horizon, check_fraction):
    if not 0.0 < check_fraction < 1.0:
        raise ValueError(f"check_fraction must lie in (0, 1), got {check_fraction}")
    gap = lookback + horizon
    usable = n_windows - gap
    n_check = int(round(usable * check_fraction))
    n_fit = usable - n_check
    if n_fit < 1 or n_check < 1:
        raise ValueError(
            f"{n_windows} validation windows cannot hold a fit segment, a gap of "
            f"{gap} windows, and a check segment with check_fraction={check_fraction}"
        )
    return dict(
        check_fraction=float(check_fraction),
        gap=gap,
        fit=np.arange(0, n_fit),
        check=np.arange(n_fit + gap, n_windows),
    )


def split_summary(split):
    return dict(
        check_fraction=split["check_fraction"],
        gap=split["gap"],
        n_fit=int(len(split["fit"])),
        n_check=int(len(split["check"])),
        check_start=int(split["check"][0]),
    )
