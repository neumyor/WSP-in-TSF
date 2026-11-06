"""Build a placebo world whose targets are the checkpoint forecast plus block-resampled residuals."""

import json
import math
import os
import shutil

import numpy as np
import torch

from .data import load_windows, split_validation
from .model import load_model, load_state, mean_losses, predict, score_state


def block_bootstrap(residual, length, block, rng):
    n = residual.shape[0]
    block = min(block, n)
    starts = rng.integers(0, n - block + 1, size=math.ceil(length / block))
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:length]
    return residual[idx]


def block_permute(residual, block, rng, return_order=False):
    blocks = [residual[i : i + block] for i in range(0, len(residual), block)]
    order = rng.permutation(len(blocks))
    out = np.concatenate([blocks[i] for i in order], axis=0)
    return (out, order.tolist()) if return_order else out


def build_placebo_world(real_base_dir, out_dir, seed, block=192, method="permutation"):
    if os.path.exists(out_dir):
        raise SystemExit(f"placebo world already exists: {out_dir}")
    model, meta = load_model(real_base_dir)
    base_state = load_state(os.path.join(real_base_dir, "base_state.pt"))
    x_val, y_val = load_windows(real_base_dir, "val")
    x_test, y_test = load_windows(real_base_dir, "test")
    horizon = int(meta["horizon"])
    split = split_validation(len(x_val), int(meta["seq_len"]), horizon, 0.5)

    pred_val = predict(model, x_val)
    pred_test = predict(model, x_test)
    val_targets = pred_val.clone()
    test_targets = pred_test.clone()
    rng = np.random.default_rng(seed)
    residuals = {}
    segments = {}
    for name, idx, y_real, pred in (
        ("fit", split["fit"], y_val, pred_val),
        ("check", split["check"], y_val, pred_val),
        ("test", np.arange(len(y_test)), y_test, pred_test),
    ):
        residual = (y_real - pred)[idx].cpu().numpy()
        residuals[name] = float(np.mean(residual**2))
        boundaries = [[i, min(i + block, len(idx))] for i in range(0, len(idx), block)]
        if method == "permutation":
            resampled, order = block_permute(residual, block, rng, return_order=True)
            segments[name] = dict(
                length=int(len(idx)),
                segment_start=int(idx[0]) if len(idx) else None,
                block=int(block),
                block_boundaries=boundaries,
                permutation=order,
            )
        elif method == "bootstrap":
            resampled = block_bootstrap(residual, len(idx), block, rng)
            segments[name] = dict(length=int(len(idx)), segment_start=int(idx[0]) if len(idx) else None, block=int(block), method="bootstrap")
        else:
            raise ValueError(f"unknown placebo method {method!r}")
        if name == "test":
            test_targets[idx] = pred[idx] + torch.from_numpy(resampled)
        else:
            val_targets[idx] = pred[idx] + torch.from_numpy(resampled)

    os.makedirs(out_dir)
    shutil.copyfile(os.path.join(real_base_dir, "base_state.pt"), os.path.join(out_dir, "base_state.pt"))
    shutil.copyfile(os.path.join(real_base_dir, "base_meta.json"), os.path.join(out_dir, "base_meta.json"))
    torch.save(dict(inputs=x_val, targets=val_targets), os.path.join(out_dir, "val_data.pt"))
    torch.save(dict(inputs=x_test, targets=test_targets), os.path.join(out_dir, "test_data.pt"))

    def segment_mse(x, y):
        mse, mae = score_state(model, base_state, x, y)
        return mean_losses(mse, mae)

    base = dict(
        fit=segment_mse(x_val[split["fit"]], val_targets[split["fit"]]),
        check=segment_mse(x_val[split["check"]], val_targets[split["check"]]),
        test=segment_mse(x_test, test_targets),
    )
    real = dict(
        fit=segment_mse(x_val[split["fit"]], y_val[split["fit"]]),
        check=segment_mse(x_val[split["check"]], y_val[split["check"]]),
        test=segment_mse(x_test, y_test),
    )
    with open(os.path.join(out_dir, "base_metrics.json"), "w") as f:
        json.dump(base, f, indent=2)
    world = dict(
        seed=int(seed),
        block=int(block),
        method=method,
        level="window",
        segments=segments,
        overlap_note=(
            "Residual blocks are permuted at the window level. Each residual is one 96-step, 7-variable target window; "
            "inputs stay unchanged, so overlapping input windows no longer share a consistent underlying target series."
        ),
        split=dict(
            n_fit=int(len(split["fit"])),
            n_check=int(len(split["check"])),
            gap=int(split["gap"]),
            check_start=int(split["check"][0]),
        ),
        placebo_base=base,
        real_base=real,
        residual_mse=residuals,
    )
    with open(os.path.join(out_dir, "placebo_world.json"), "w") as f:
        json.dump(world, f, indent=2)
    return world
