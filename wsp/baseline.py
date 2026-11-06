"""Gradient fine-tuning on the fit segment, reported next to the agent as a comparison."""

import json
import os

import torch

from .data import load_windows, split_summary, split_validation
from .model import clone_state, forecast, load_model, mean_losses, predict, score_state
from .stats import paired_gain


BASELINE_FILE = "gradient_baseline.json"
BASELINE_STATE = "gradient_baseline.pt"


def fine_tune(
    model,
    state,
    x,
    y,
    *,
    prefix,
    lr,
    steps,
    batch_size=256,
    seed=0,
    check=None,
    eval_every=20,
    patience=10,
):
    """Adam on parameters named `prefix*`; with `check`, return the lowest-check-MSE state including step 0."""
    model.load_state_dict(state, strict=True)
    names = [name for name, _ in model.named_parameters() if name.startswith(prefix)]
    if not names:
        raise ValueError(f"no parameters match prefix {prefix!r}")
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith(prefix))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    generator = torch.Generator().manual_seed(seed)

    def check_mse():
        return float(((predict(model, check[0]) - check[1]) ** 2).mean())

    best_state, best_step = clone_state(state), 0
    best_mse = check_mse() if check is not None else None
    stale = 0
    try:
        for step in range(1, steps + 1):
            model.train()
            idx = torch.randint(0, len(x), (min(batch_size, len(x)),), generator=generator)
            loss = ((forecast(model, x[idx]) - y[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if check is None or step % eval_every:
                continue
            current = check_mse()
            if current < best_mse:
                best_state, best_step, best_mse, stale = clone_state(model.state_dict()), step, current, 0
            else:
                stale += 1
                if stale >= patience:
                    break
    finally:
        for param in model.parameters():
            param.requires_grad_(True)
        model.eval()
    if check is None:
        return clone_state(model.state_dict()), steps, None
    return best_state, best_step, best_mse


def run_gradient_baseline(base_dir, check_fraction, scopes, lrs, steps, eval_every, patience, seed=0):
    model, meta = load_model(base_dir)
    base_state = clone_state(model.state_dict())
    horizon = int(meta["horizon"])
    block = 2 * horizon

    x_val, y_val = load_windows(base_dir, "val")
    x_test, y_test = load_windows(base_dir, "test")
    split = split_validation(len(x_val), int(meta["seq_len"]), horizon, check_fraction)
    x_fit, y_fit = x_val[split["fit"]], y_val[split["fit"]]
    x_check, y_check = x_val[split["check"]], y_val[split["check"]]

    base_fit = score_state(model, base_state, x_fit, y_fit)
    base_check = score_state(model, base_state, x_check, y_check)
    base_test = score_state(model, base_state, x_test, y_test)

    best = dict(scope="none", prefix=None, lr=None, step=0, check_mse=float(base_check[0].mean()), state=base_state)
    configs = []
    for scope, prefix in scopes.items():
        for lr in lrs:
            state, step, check_mse = fine_tune(
                model,
                base_state,
                x_fit,
                y_fit,
                prefix=prefix,
                lr=lr,
                steps=steps,
                seed=seed,
                check=(x_check, y_check),
                eval_every=eval_every,
                patience=patience,
            )
            configs.append(dict(scope=scope, prefix=prefix, lr=lr, step=step, check_mse=check_mse))
            print(f"gradient baseline {scope:>4} lr={lr:g}: best step {step}, check MSE {check_mse:.6f}")
            if check_mse < best["check_mse"]:
                best = dict(scope=scope, prefix=prefix, lr=lr, step=step, check_mse=check_mse, state=state)

    tuned_check = score_state(model, best["state"], x_check, y_check)
    tuned_test = score_state(model, best["state"], x_test, y_test)
    torch.save(best["state"], os.path.join(base_dir, BASELINE_STATE))
    result = dict(
        split=split_summary(split),
        block=block,
        base=dict(fit=mean_losses(*base_fit), check=mean_losses(*base_check), test=mean_losses(*base_test)),
        configs=configs,
        selected=dict(
            scope=best["scope"],
            prefix=best["prefix"],
            lr=best["lr"],
            step=best["step"],
            check=mean_losses(*tuned_check),
            test=mean_losses(*tuned_test),
            check_gain=paired_gain(base_check[0], tuned_check[0], block),
            test_gain=paired_gain(base_test[0], tuned_test[0], block),
        ),
    )
    with open(os.path.join(base_dir, BASELINE_FILE), "w") as f:
        json.dump(result, f, indent=2)
    return result


def load_gradient_baseline(base_dir, split):
    """Return the stored baseline when it exists and used the same fit/check split, else None."""
    path = os.path.join(base_dir, BASELINE_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        result = json.load(f)
    if result["split"] != split_summary(split):
        return None
    result["state_path"] = os.path.join(base_dir, BASELINE_STATE)
    return result
