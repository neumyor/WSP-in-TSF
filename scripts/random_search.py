"""Matched random search for the seven agent runs that changed weights."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from wsp.data import load_windows, split_validation
from wsp.model import load_model, load_state, predict, window_losses
from wsp.stats import paired_gain


OBJECTS = [
    ("deepseek_run4", "deepseek-v4.1-flash-mb", "runs/wsp_h96_deepseek_run4"),
    ("deepseek_run5", "deepseek-v4.1-flash-mb", "runs/wsp_h96_deepseek_run5"),
    ("deepseek_run6", "deepseek-v4.1-flash-mb", "runs/wsp_h96_deepseek_run6"),
    ("glm_run1", "glm-5.3-mb", "runs/wsp_h96_glm53_run1"),
    ("glm_run2", "glm-5.3-mb", "runs/wsp_h96_glm53_run2"),
    ("glm_run3", "glm-5.3-mb", "runs/wsp_h96_glm53_run3"),
    ("gpt_run3", "gpt-6.1-sol", "runs/wsp_h96_gpt61_run3"),
]


def flat_state(state):
    return torch.cat([value.float().reshape(-1) for value in state.values()])


def set_flat(model, flat):
    torch.nn.utils.vector_to_parameters(flat, model.parameters())


def loss(model, x, y):
    return float(((predict(model, x) - y) ** 2).mean())


def percentage(absolute, base_mse):
    return 100.0 * absolute / base_mse


def pct_ci(gain, base_mse):
    if gain["ci"] is None:
        return None
    return [percentage(gain["ci"][0], base_mse), percentage(gain["ci"][1], base_mse)]


def object_stats(base_dir, run_dir):
    model, meta = load_model(base_dir)
    base_state = load_state(os.path.join(base_dir, "base_state.pt"))
    edited = load_state(os.path.join(run_dir, "submission.pt"))
    names = list(base_state)
    base_flat = flat_state(base_state)
    edited_flat = flat_state(edited)
    delta = edited_flat - base_flat
    changed = delta.abs() > 1e-4
    k = int(changed.sum())
    l2 = float(delta.norm())
    rms = float(torch.sqrt((delta[changed] ** 2).mean())) if k else 0.0
    edited_tensors, offset = [], 0
    for name in names:
        count = base_state[name].numel()
        if (delta[offset : offset + count].abs() > 1e-4).any():
            edited_tensors.append(name)
        offset += count
    return model, meta, base_state, base_flat, k, l2, rms, edited_tensors


def run_replicate(model, x_fit, y_fit, x_check, y_check, x_test, y_test, base_flat, base_losses, pool, k, l2, rms, seed, budget=100, checks=10, coordinate_set="all", bootstrap_reps=1000, step_scale=1.0):
    rng = np.random.default_rng(seed)
    coords = torch.as_tensor(rng.choice(pool, size=min(k, len(pool)), replace=False), dtype=torch.long)
    fit_subset = rng.choice(len(x_fit), size=min(512, len(x_fit)), replace=False)
    subset_x, subset_y = x_fit[fit_subset], y_fit[fit_subset]
    delta = torch.zeros_like(base_flat)
    current = loss(model, subset_x, subset_y)
    candidates = [dict(step=0, check=base_losses["check"], delta=delta.clone(), check_state=True)]
    accepted = 0
    for step in range(1, budget + 1):
        proposal = delta.clone()
        proposal[coords] += torch.from_numpy(rng.normal(0.0, rms * step_scale, size=len(coords)).astype(np.float32))
        norm = float(proposal.norm())
        if norm > l2 > 0:
            proposal *= l2 / norm
        set_flat(model, base_flat + proposal)
        candidate_loss = loss(model, subset_x, subset_y)
        if candidate_loss < current:
            delta, current, accepted = proposal, candidate_loss, accepted + 1
        if step % (budget // checks) == 0:
            set_flat(model, base_flat + delta)
            check_mse = loss(model, x_check, y_check)
            candidates.append(dict(step=step, check=check_mse, delta=delta.clone()))
    best = min(candidates, key=lambda row: row["check"])
    set_flat(model, base_flat + best["delta"])
    fit_pred = predict(model, x_fit)
    check_pred = predict(model, x_check)
    test_pred = predict(model, x_test)
    fit_mse = window_losses(fit_pred, y_fit)[0]
    check_mse = window_losses(check_pred, y_check)[0]
    test_mse = window_losses(test_pred, y_test)[0]
    gains = {
        "fit": paired_gain(base_losses["fit_windows"], fit_mse, 192, reps=bootstrap_reps),
        "check": paired_gain(base_losses["check_windows"], check_mse, 192, reps=bootstrap_reps),
        "test": paired_gain(base_losses["test_windows"], test_mse, 192, reps=bootstrap_reps),
    }
    return dict(
        seed=int(seed),
        l2_cap=float(l2),
        rms=float(rms),
        coordinate_set=coordinate_set,
        step_scale=float(step_scale),
        k=int(len(coords)),
        fit_mse=float(fit_mse.mean()),
        check_mse=float(check_mse.mean()),
        test_mse=float(test_mse.mean()),
        fit_gain_pct=percentage(gains["fit"]["gain"], base_losses["fit"]),
        check_gain_pct=percentage(gains["check"]["gain"], base_losses["check"]),
        test_gain_pct=percentage(gains["test"]["gain"], base_losses["test"]),
        fit_gain_pct_ci=pct_ci(gains["fit"], base_losses["fit"]),
        check_gain_pct_ci=pct_ci(gains["check"], base_losses["check"]),
        test_gain_pct_ci=pct_ci(gains["test"], base_losses["test"]),
        selected_step=int(best["step"]),
        accepted_steps=int(accepted),
        forward_used=int(budget),
        checks_used=int(checks),
    )


def coord_indices(names, base_state, kind, edited_tensors):
    offsets = np.cumsum([0] + [base_state[n].numel() for n in names])
    if kind == "all":
        return np.arange(offsets[-1])
    indices = []
    for name in edited_tensors:
        start = int(offsets[names.index(name)])
        indices.extend(range(start, start + base_state[name].numel()))
    return np.array(indices)


def summarize(rows, agent_test_pct, agent_check_pct):
    def stats(values):
        values = np.array(values, dtype=float)
        return dict(
            mean=float(values.mean()),
            std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            p5=float(np.percentile(values, 5)),
            p95=float(np.percentile(values, 95)),
        )

    test = [row["test_gain_pct"] for row in rows]
    higher = sum(value > agent_test_pct for value in test)
    changed = [row for row in rows if row["selected_step"] > 0]
    return dict(
        n=len(rows),
        fit=stats([row["fit_gain_pct"] for row in rows]),
        check=stats([row["check_gain_pct"] for row in rows]),
        test=stats(test),
        agent_test_pct=float(agent_test_pct),
        agent_check_pct=float(agent_check_pct),
        reps_with_higher_test_gain=higher,
        fraction_higher_than_agent=higher / len(rows),
        selected_changed=len(changed),
        selected_changed_check_mean=float(np.mean([row["check_gain_pct"] for row in changed])) if changed else None,
        selected_changed_test_mean=float(np.mean([row["test_gain_pct"] for row in changed])) if changed else None,
        accepted_steps_mean=float(np.mean([row["accepted_steps"] for row in rows])),
        step_norm_over_cap=float(rows[0]["step_scale"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="runs/base_etth1_h96")
    parser.add_argument("--out", default="runs/controls/random_search")
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--checks", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=20261106)
    parser.add_argument("--coordinate-sets", default="all,edited")
    parser.add_argument("--objects", default="", help="comma-separated object ids; empty means all")
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--step-scales", default="1.0", help="comma-separated step scales in units of the L2 cap")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model, meta = load_model(args.base_dir)
    x_val, y_val = load_windows(args.base_dir, "val")
    x_test, y_test = load_windows(args.base_dir, "test")
    split = split_validation(len(x_val), int(meta["seq_len"]), int(meta["horizon"]), 0.5)
    x_fit, y_fit = x_val[split["fit"]], y_val[split["fit"]]
    x_check, y_check = x_val[split["check"]], y_val[split["check"]]

    all_rows = []
    summaries = []
    wanted = {item.strip() for item in args.objects.split(",") if item.strip()}
    objects = [item for item in OBJECTS if not wanted or item[0] in wanted]
    for object_index, (object_id, model_name, run_dir) in enumerate(objects):
        model, meta, base_state, base_flat, k, l2, rms, edited_tensors = object_stats(args.base_dir, run_dir)
        set_flat(model, base_flat)
        base_losses = dict(
            fit=float(((predict(model, x_fit) - y_fit) ** 2).mean()),
            check=float(((predict(model, x_check) - y_check) ** 2).mean()),
            test=float(((predict(model, x_test) - y_test) ** 2).mean()),
            fit_windows=window_losses(predict(model, x_fit), y_fit)[0],
            check_windows=window_losses(predict(model, x_check), y_check)[0],
            test_windows=window_losses(predict(model, x_test), y_test)[0],
        )
        with open(os.path.join(run_dir, "final_report.json")) as f:
            report = json.load(f)
        agent_test_pct = percentage(report["metrics"]["test"]["gain"]["gain"], report["metrics"]["test"]["base"]["mse"])
        agent_check_pct = percentage(report["metrics"]["check"]["gain"]["gain"], report["metrics"]["check"]["base"]["mse"])
        names = list(base_state)
        for kind in [item.strip() for item in args.coordinate_sets.split(",") if item.strip()]:
            pool = coord_indices(names, base_state, kind, edited_tensors)
            for scale in [float(item) for item in args.step_scales.split(",") if item.strip()]:
                rows = []
                for replicate in range(1, args.replicates + 1):
                    seed = args.seed_base + object_index * 1000 + replicate
                    row = run_replicate(
                        model, x_fit, y_fit, x_check, y_check, x_test, y_test,
                        base_flat, base_losses, pool, k, l2, rms, seed,
                        budget=args.budget, checks=args.checks, coordinate_set=kind,
                        bootstrap_reps=args.bootstrap_reps, step_scale=scale,
                    )
                    row.update(object=object_id, model=model_name, coordinate_set=kind)
                    rows.append(row)
                    all_rows.append(row)
                    if replicate % 10 == 0:
                        print(f"  {object_id} {kind} step={scale}: {replicate}/{args.replicates} mean test {np.mean([r['test_gain_pct'] for r in rows]):+.3f}%", flush=True)
                summary = summarize(rows, agent_test_pct, agent_check_pct)
                summary.update(object=object_id, model=model_name, coordinate_set=kind, step_scale=scale, k=k, l2_cap=l2, rms=rms, edited_tensors=edited_tensors)
                summaries.append(summary)
                print(f"{object_id} {kind} step={scale}: mean test {summary['test']['mean']:+.3f}%, agent {agent_test_pct:+.3f}%, fraction higher {summary['fraction_higher_than_agent']:.2f}", flush=True)

    with open(os.path.join(args.out, "replicates.json"), "w") as f:
        json.dump(all_rows, f, indent=2)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    lines = [
        "# Matched random search",
        "",
        "Each replicate fixes k coordinates, takes 100 Gaussian steps evaluated on 512 fit windows, "
        "accepts improving steps, caps the L2 norm at the agent's norm, and selects among 10 check "
        "candidates plus the unedited checkpoint. Gains are percent of the unedited MSE.",
        "",
        "| object | coords | step | k | mean test | std | p5 | p95 | agent test | fraction higher than agent | selected changed | changed check | changed test | accepted steps | step norm / L2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        changed_check = "-" if s["selected_changed_check_mean"] is None else f"{s['selected_changed_check_mean']:+.3f}%"
        changed_test = "-" if s["selected_changed_test_mean"] is None else f"{s['selected_changed_test_mean']:+.3f}%"
        lines.append(
            f"| {s['object']} | {s['coordinate_set']} | {s['step_scale']:.3f} | {s['k']} | {s['test']['mean']:+.3f}% | {s['test']['std']:.3f} | "
            f"{s['test']['p5']:+.3f}% | {s['test']['p95']:+.3f}% | {s['agent_test_pct']:+.3f}% | {s['fraction_higher_than_agent']:.3f} | "
            f"{s['selected_changed']} | {changed_check} | {changed_test} | "
            f"{s['accepted_steps_mean']:.2f} | {s['step_norm_over_cap']:.3f} |"
        )
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
