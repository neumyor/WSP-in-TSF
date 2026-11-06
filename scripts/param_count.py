"""Print parameter counts for candidate phase-only PhaseFormer configs."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.PhaseFormer import PhaseFormer
from src.models.phaseformer_presets import PhaseFormerPresetConfig, make_exp_args


CANDIDATES = [
    dict(name="d8_M8_pos1", latent_dim=8, phase_num_routers=8, phase_use_pos_embed=True),
    dict(name="d8_M16_pos1", latent_dim=8, phase_num_routers=16, phase_use_pos_embed=True),
    dict(name="d8_M8_pos0", latent_dim=8, phase_num_routers=8, phase_use_pos_embed=False),
    dict(name="d8_M16_pos0", latent_dim=8, phase_num_routers=16, phase_use_pos_embed=False),
    dict(name="d4_M8_pos1", latent_dim=4, phase_num_routers=8, phase_use_pos_embed=True),
    dict(name="d16_M8_pos1", latent_dim=16, phase_num_routers=8, phase_use_pos_embed=True),
]


def build(candidate, layers, lookback, horizon):
    hyperparams = dict(
        scheme_name="param_count",
        layers=layers,
        latent_dim=candidate["latent_dim"],
        phase_encoder_hidden=32,
        predictor_hidden=64,
        phase_num_routers=candidate["phase_num_routers"],
        phase_attn_heads=1,
        phase_attn_dropout=0.0,
        phase_use_pos_embed=candidate["phase_use_pos_embed"],
        phase_encoder_use_mlp=False,
        predictor_use_mlp=False,
        use_residual_head=False,
        learning_rate=1e-3,
        train_epochs=1,
        patience=1,
        seed=2021,
        loss_func="mse",
        use_huber_loss=False,
    )
    exp_args = make_exp_args("ETTh1", lookback, horizon, hyperparams)
    config = PhaseFormerPresetConfig(exp_args, lookback, horizon, hyperparams)
    model = PhaseFormer(config)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=720)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    rows = []
    for layers in args.layers:
        for candidate in CANDIDATES:
            model = build(candidate, layers, args.lookback, args.horizon)
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            rows.append(
                dict(name=candidate["name"], layers=layers, total=total, trainable=trainable)
            )
            print(f"{candidate['name']:>12} layers={layers} total={total} trainable={trainable}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
