"""Build PhaseFormer from exported metadata and run batched forecasts."""

import json
import os

import numpy as np
import torch

from src.models.PhaseFormer import PhaseFormer
from src.models.phaseformer_presets import PhaseFormerPresetConfig, make_exp_args


def build_model(meta):
    hyperparams = dict(meta["hyperparams"])
    exp_args = make_exp_args(meta["dataset"], meta["seq_len"], meta["horizon"], hyperparams)
    exp_args.dataset_args.num_workers = 0
    exp_args.training_args.num_workers = 0
    config = PhaseFormerPresetConfig(exp_args, meta["seq_len"], meta["horizon"], hyperparams)
    return PhaseFormer(config)


def read_meta(base_dir):
    path = os.path.join(base_dir, "base_meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing checkpoint metadata: {path}")
    with open(path) as f:
        return json.load(f)


def load_state(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing weight file: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def clone_state(state):
    return {name: value.detach().clone() for name, value in state.items()}


def load_model(base_dir):
    meta = read_meta(base_dir)
    model = build_model(meta)
    model.load_state_dict(load_state(os.path.join(base_dir, "base_state.pt")), strict=True)
    model.eval()
    return model, meta


def forecast(model, x):
    out = model(x_enc=x)
    if isinstance(out, tuple):
        out = out[0]
    return out[:, -model.pred_len :, :]


@torch.no_grad()
def predict(model, x, batch_size=512):
    model.eval()
    return torch.cat([forecast(model, x[i : i + batch_size]) for i in range(0, len(x), batch_size)])


def window_losses(pred, target):
    diff = pred - target
    return (diff**2).mean(dim=(1, 2)).numpy(), diff.abs().mean(dim=(1, 2)).numpy()


def score_state(model, state, x, y):
    model.load_state_dict(state, strict=True)
    mse, mae = window_losses(predict(model, x), y)
    return mse, mae


def mean_losses(mse, mae):
    return dict(mse=float(np.mean(mse)), mae=float(np.mean(mae)))
