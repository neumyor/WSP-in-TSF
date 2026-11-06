"""Diagnose a WSP run: decoder-subspace fits, gradient fine-tuning comparisons, and temporal transfer of the submitted edit."""

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from wsp.baseline import fine_tune
from wsp.data import load_windows, split_validation
from wsp.model import clone_state, load_model, load_state, predict, score_state
from wsp.stats import block_bootstrap_ci


DECODER = ["predictor.decoder.weight", "predictor.decoder.bias"]


@torch.no_grad()
def decoder_jacobian(model, x):
    """The output is affine in the 28 decoder parameters, so the design matrix is exact."""
    params = dict(model.named_parameters())
    saved = {n: params[n].detach().clone() for n in DECODER}
    for n in DECODER:
        params[n].zero_()
    base = predict(model, x)
    columns = []
    for n in DECODER:
        flat = params[n].view(-1)
        for j in range(flat.numel()):
            flat[j] = 1.0
            columns.append((predict(model, x) - base).reshape(-1))
            flat[j] = 0.0
    for n in DECODER:
        params[n].copy_(saved[n])
    return base.reshape(-1), torch.stack(columns, dim=1)


def fit_decoder(model, state, x, y, ridge=1e-6):
    model.load_state_dict(state, strict=True)
    base, design = decoder_jacobian(model, x)
    gram = design.T @ design + ridge * torch.eye(design.shape[1])
    theta = torch.linalg.solve(gram, design.T @ (y.reshape(-1) - base))
    fitted = copy.deepcopy(state)
    w_numel = fitted[DECODER[0]].numel()
    fitted[DECODER[0]] = theta[:w_numel].reshape(fitted[DECODER[0]].shape)
    fitted[DECODER[1]] = theta[w_numel:].reshape(fitted[DECODER[1]].shape)
    return fitted


def load_train_windows(meta, stride=4):
    from src.dataset.data_factory import data_provider
    from src.models.phaseformer_presets import make_exp_args

    horizon = int(meta["horizon"])
    exp_args = make_exp_args(meta["dataset"], meta["seq_len"], horizon, dict(meta["hyperparams"]))
    dataset_args = copy.deepcopy(exp_args.dataset_args)
    dataset_args.num_workers = 0
    dataset, _ = data_provider(dataset_args, "train")
    xs, ys = [], []
    for i in range(0, len(dataset), stride):
        item = dataset[i]
        xs.append(torch.as_tensor(item[0]).float())
        ys.append(torch.as_tensor(item[1]).float()[-horizon:])
    return torch.stack(xs), torch.stack(ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="runs/base_etth1_h96")
    parser.add_argument("--run-dir", default="runs/wsp_etth1_h96")
    parser.add_argument("--check-fraction", type=float, default=0.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model, meta = load_model(args.base_dir)
    horizon = int(meta["horizon"])
    block = 2 * horizon
    base_state = clone_state(model.state_dict())
    edited_state = load_state(os.path.join(args.run_dir, "submission.pt"))

    xv, yv = load_windows(args.base_dir, "val")
    xt, yt = load_windows(args.base_dir, "test")
    xtr, ytr = load_train_windows(meta)

    split = split_validation(len(xv), int(meta["seq_len"]), horizon, args.check_fraction)
    first, second = torch.as_tensor(split["fit"]), torch.as_tensor(split["check"])
    splits = {
        "val": (xv, yv),
        "val_first": (xv[first], yv[first]),
        "val_second": (xv[second], yv[second]),
        "test": (xt, yt),
    }

    states = {"base": base_state, "agent_edit": edited_state}
    states["decoder_ls_full_val"] = fit_decoder(model, base_state, xv, yv)
    states["decoder_ls_first_half"] = fit_decoder(model, base_state, xv[first], yv[first])
    states["decoder_ls_train"] = fit_decoder(model, base_state, xtr, ytr)
    states["adam_all_first_half"] = fine_tune(model, base_state, xv[first], yv[first], prefix="", lr=1e-3, steps=300)[0]
    states["adam_all_full_val"] = fine_tune(model, base_state, xv, yv, prefix="", lr=1e-3, steps=300)[0]

    def losses(state):
        return {name: score_state(model, state, x, y)[0] for name, (x, y) in splits.items()}

    base_losses = losses(base_state)
    results = {}
    for name, state in states.items():
        current = losses(state)
        row = {}
        for split in splits:
            gain, ci = block_bootstrap_ci(base_losses[split] - current[split], block)
            row[split] = dict(mse=float(current[split].mean()), gain=gain, gain_ci=ci)
        results[name] = row

    distribution = {}
    for split, (x, y) in {"train": (xtr, ytr), "val": (xv, yv), "test": (xt, yt)}.items():
        distribution[split] = dict(
            target_mean=y.mean(dim=(0, 1)).tolist(),
            target_std=y.std(dim=(0, 1)).tolist(),
            input_window_std=x.std(dim=1).mean(dim=0).tolist(),
        )

    edited_losses = losses(edited_state)
    blocks = []
    for split in ("val", "test"):
        diff = base_losses[split] - edited_losses[split]
        for k, chunk in enumerate(np.array_split(np.arange(len(diff)), 8)):
            blocks.append(dict(split=split, block=k, base=float(base_losses[split][chunk].mean()), gain=float(diff[chunk].mean())))

    out = args.out or os.path.join(args.run_dir, "diagnosis.json")
    with open(out, "w") as f:
        json.dump(dict(results=results, distribution=distribution, edit_gain_by_block=blocks), f, indent=2)

    print(f"{'state':>24} | " + " | ".join(f"{s:>22}" for s in splits))
    for name, row in results.items():
        cells = [f"{row[s]['mse']:.4f} ({row[s]['gain']:+.4f})" for s in splits]
        print(f"{name:>24} | " + " | ".join(f"{c:>22}" for c in cells))
    print("\nedit gain by time block (base MSE, gain):")
    for b in blocks:
        print(f"  {b['split']:>4} block {b['block']}: base {b['base']:.4f} gain {b['gain']:+.4f}")
    print("\ntarget std per variable:")
    for split, row in distribution.items():
        print(f"  {split:>5}: " + " ".join(f"{v:.3f}" for v in row["target_std"]))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
