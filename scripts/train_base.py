"""Train a phase-only PhaseFormer base checkpoint on ETTh1 and export artifacts."""

import argparse
import json
import os
import sys
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from src.dataset.data_factory import data_provider
from src.models.PhaseFormer import PhaseFormer
from src.models.phaseformer_presets import PhaseFormerPresetConfig, make_exp_args
from src.training.runner import build_logger, build_trainer, restore_best_checkpoint


PHASE_ONLY = dict(
    scheme_name="wsp_phase_only",
    layers=1,
    latent_dim=6,
    phase_encoder_hidden=32,
    predictor_hidden=64,
    phase_num_routers=16,
    phase_attn_heads=1,
    phase_attn_dropout=0.0,
    phase_attn_use_relpos=True,
    phase_use_pos_embed=True,
    phase_encoder_use_mlp=False,
    predictor_use_mlp=False,
    use_residual_head=False,
    learning_rate=1e-3,
    train_epochs=30,
    patience=8,
    seed=2021,
    loss_func="mse",
    use_huber_loss=False,
)


def build_experiment(args):
    hyperparams = deepcopy(PHASE_ONLY)
    hyperparams.update(
        dict(
            layers=args.layers,
            latent_dim=args.latent_dim,
            phase_num_routers=args.routers,
            learning_rate=args.lr,
            train_epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
        )
    )
    exp_args = make_exp_args(
        args.dataset,
        args.seq_len,
        args.horizon,
        hyperparams,
        batch_size=args.batch_size,
    )
    exp_args.dataset_args.num_workers = 0
    exp_args.training_args.num_workers = 0
    config = PhaseFormerPresetConfig(exp_args, args.seq_len, args.horizon, hyperparams)
    return hyperparams, exp_args, config


def make_loaders(exp_args, batch_size):
    eval_args = deepcopy(exp_args.dataset_args)
    eval_args.batch_size = batch_size
    train_set, train_loader = data_provider(eval_args, "train")
    _, val_loader = data_provider(eval_args, "val")
    _, test_loader = data_provider(eval_args, "test")
    if hasattr(train_set, "data_stamp"):
        return train_loader, val_loader, test_loader, int(train_set.data_stamp.shape[-1])
    return train_loader, val_loader, test_loader, None


@torch.no_grad()
def collect_predictions(model, loader, horizon):
    model.eval()
    device = next(model.parameters()).device
    inputs, targets, preds = [], [], []
    for batch in loader:
        x = batch[0].float().to(device)
        y = batch[1].float().to(device)
        out = model(x_enc=x)
        if isinstance(out, tuple):
            out = out[0]
        pred = out[:, -horizon:, :]
        target = y[:, -horizon:, :]
        inputs.append(x.detach().cpu().float())
        targets.append(target.detach().cpu().float())
        preds.append(pred.detach().cpu().float())
    return (
        torch.cat(inputs, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(preds, dim=0),
    )


def metrics_from_predictions(pred, target):
    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    sq = (pred - target) ** 2
    per_horizon = sq.mean(dim=(0, 2)).tolist()
    per_variable = sq.mean(dim=(0, 1)).tolist()
    return dict(mse=mse, mae=mae, per_horizon_mse=per_horizon, per_variable_mse=per_variable)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--seq-len", type=int, default=720)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--routers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--out", default="runs/base_etth1_h96")
    parser.add_argument("--log-dir", default="runs/lightning_logs")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(args.seed, workers=True)

    hyperparams, exp_args, model_config = build_experiment(args)
    model = PhaseFormer(model_config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {num_params}")

    train_loader, val_loader, test_loader, time_mark_dim = make_loaders(
        exp_args, args.batch_size
    )
    if time_mark_dim is not None:
        hyperparams["time_mark_dim"] = time_mark_dim
        model_config = PhaseFormerPresetConfig(
            exp_args, args.seq_len, args.horizon, hyperparams
        )
        model = PhaseFormer(model_config)

    os.makedirs(args.out, exist_ok=True)
    logger = build_logger(args.log_dir, name="PhaseFormer", version=os.path.basename(args.out))
    trainer, checkpoint = build_trainer(
        max_epochs=args.epochs,
        logger=logger,
        patience=args.patience,
        progress=args.progress,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    restore_best_checkpoint(model, checkpoint)

    eval_train_loader, eval_val_loader, eval_test_loader, _ = make_loaders(
        exp_args, args.eval_batch_size
    )
    val_x, val_y, val_pred = collect_predictions(model, eval_val_loader, args.horizon)
    test_x, test_y, test_pred = collect_predictions(model, eval_test_loader, args.horizon)

    val_metrics = metrics_from_predictions(val_pred, val_y)
    test_metrics = metrics_from_predictions(test_pred, test_y)

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, os.path.join(args.out, "base_state.pt"))
    torch.save({"inputs": val_x, "targets": val_y}, os.path.join(args.out, "val_data.pt"))
    torch.save({"inputs": test_x, "targets": test_y}, os.path.join(args.out, "test_data.pt"))

    meta = dict(
        dataset=args.dataset,
        seq_len=args.seq_len,
        horizon=args.horizon,
        period_len=int(getattr(model_config, "period_len", 24)),
        enc_in=int(getattr(model_config, "enc_in", 7)),
        seed=args.seed,
        num_params=num_params,
        hyperparams=hyperparams,
        checkpoint=checkpoint.best_model_path,
        best_val_loss=float(checkpoint.best_model_score or float("nan")),
        epochs_completed=int(trainer.current_epoch),
    )
    with open(os.path.join(args.out, "base_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(args.out, "base_metrics.json"), "w") as f:
        json.dump(
            dict(
                val=val_metrics,
                test=test_metrics,
                num_val_windows=int(val_x.shape[0]),
                num_test_windows=int(test_x.shape[0]),
            ),
            f,
            indent=2,
        )

    print(
        f"val MSE={val_metrics['mse']:.6f} MAE={val_metrics['mae']:.6f} | "
        f"test MSE={test_metrics['mse']:.6f} MAE={test_metrics['mae']:.6f}"
    )
    print(f"artifacts written to {args.out}")


if __name__ == "__main__":
    main()
