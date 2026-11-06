"""Run the agent and gradient baseline in placebo worlds and compare them."""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wsp.data import load_windows
from wsp.model import load_model, load_state, mean_losses, score_state
from wsp.placebo import build_placebo_world
from wsp.stats import paired_gain


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHORT = {
    "gpt-6.1-sol": "gpt61",
    "deepseek-v4.1-flash-mb": "deepseek41",
    "glm-5.3-mb": "glm53",
}
OFFSET = {"gpt-6.1-sol": 0, "deepseek-v4.1-flash-mb": 100, "glm-5.3-mb": 200}


def pct(gain, base_mse):
    return 100.0 * gain / base_mse


def fmt_gain(gain, base_mse):
    text = f"{pct(gain['gain'], base_mse):+.3f}%"
    if gain["ci"]:
        text += f" [{pct(gain['ci'][0], base_mse):+.3f}, {pct(gain['ci'][1], base_mse):+.3f}]"
    return text


def fmt(row):
    return fmt_gain(row["gain"], row["base"]["mse"])


def real_test_gain(real_base_dir, agent_dir):
    model, meta = load_model(real_base_dir)
    base = load_state(os.path.join(real_base_dir, "base_state.pt"))
    submitted = load_state(os.path.join(agent_dir, "submission.pt"))
    x, y = load_windows(real_base_dir, "test")
    base_mse, base_mae = score_state(model, base, x, y)
    sub_mse, sub_mae = score_state(model, submitted, x, y)
    gain = paired_gain(base_mse, sub_mse, 2 * int(meta["horizon"]))
    return dict(
        base=mean_losses(base_mse, base_mae),
        submitted=mean_losses(sub_mse, sub_mae),
        gain=gain,
    )


def run_one(model_name, index, args):
    short = SHORT[model_name]
    world_dir = os.path.join(args.out, f"{short}_{index:02d}")
    base_dir = os.path.join(world_dir, "base")
    agent_dir = os.path.join(world_dir, "agent")
    seed = args.seed_base + OFFSET[model_name] + index
    if os.path.exists(os.path.join(agent_dir, "final_report.json")):
        with open(os.path.join(base_dir, "placebo_world.json")) as f:
            world = json.load(f)
        with open(os.path.join(agent_dir, "final_report.json")) as f:
            report = json.load(f)
        with open(os.path.join(agent_dir, "driver_meta.json")) as f:
            driver = json.load(f)
        return dict(
            model=model_name, index=index, seed=seed, world=world,
            placebo=report["metrics"], gradient_baseline=report.get("gradient_baseline"),
            real_test=real_test_gain(os.path.abspath(args.base_dir), agent_dir),
            candidate=report["submission"]["candidate"], submission=report["submission"],
            footprint=report["footprint"], driver=driver, world_dir=world_dir,
        )
    world = build_placebo_world(os.path.abspath(args.base_dir), base_dir, seed, args.block, args.method)

    baseline = subprocess.run(
        [sys.executable, os.path.join(REPO, "run_wsp.py"), "baseline", "--base-dir", base_dir],
        cwd=REPO, capture_output=True, text=True,
    )
    with open(os.path.join(world_dir, "baseline.log"), "w") as f:
        f.write(baseline.stdout + baseline.stderr)
    if baseline.returncode:
        raise SystemExit(f"baseline failed for {world_dir}; see baseline.log")

    agent = subprocess.run(
        [
            sys.executable, os.path.join(REPO, "run_wsp.py"), "agent",
            "--base-dir", base_dir, "--run-dir", agent_dir, "--model", model_name,
            "--budget", str(args.budget), "--max-checks", str(args.max_checks),
            "--max-turns", str(args.max_turns), "--api-timeout", str(args.api_timeout),
            "--max-tokens", str(args.max_tokens), "--context-chars", str(args.context_chars),
        ],
        cwd=REPO, capture_output=True, text=True,
    )
    with open(os.path.join(world_dir, "agent.log"), "w") as f:
        f.write(agent.stdout + agent.stderr)
    if agent.returncode:
        raise SystemExit(f"agent failed for {world_dir}; see agent.log")

    with open(os.path.join(agent_dir, "final_report.json")) as f:
        report = json.load(f)
    with open(os.path.join(agent_dir, "driver_meta.json")) as f:
        driver = json.load(f)
    real = real_test_gain(os.path.abspath(args.base_dir), agent_dir)
    return dict(
        model=model_name,
        index=index,
        seed=seed,
        world=world,
        placebo=report["metrics"],
        gradient_baseline=report.get("gradient_baseline"),
        real_test=real,
        candidate=report["submission"]["candidate"],
        submission=report["submission"],
        footprint=report["footprint"],
        driver=driver,
        world_dir=world_dir,
    )


def write_comparison(records, out_dir):
    lines = [
        "# Placebo controls",
        "",
        "Placebo targets are the unedited checkpoint forecast plus block-resampled residuals. "
        f"Method: {records[0]['world']['method']} with blocks of {records[0]['world']['block']} windows. "
        "Gains are relative to the unedited checkpoint in the same placebo world.",
        "",
        "| model | run | seed | fit gain | check gain | test gain | gradient check gain | gradient test gain | real test gain | changed | fwd | checks | tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        p = r["placebo"]
        lines.append(
            f"| {r['model']} | {r['index']} | {r['seed']} | {fmt(p['fit'])} | {fmt(p['check'])} | "
            f"{fmt(p['test'])} | {fmt_gain(p['check']['gradient_baseline_gain'], p['check']['base']['mse'])} | "
            f"{fmt_gain(p['test']['gradient_baseline_gain'], p['test']['base']['mse'])} | {fmt(r['real_test'])} | "
            f"{r['footprint']['changed']} | {r['driver']['forward_used']} | {r['driver']['checks_used']} | {r['driver']['usage']['total_tokens']} |"
        )
    lines += ["", "## Sanity checks", "", "| model | run | placebo fit / check / test MSE | real fit / check / test MSE |", "|---|---|---:|---:|"]
    for r in records:
        p, real = r["world"]["placebo_base"], r["world"]["real_base"]
        lines.append(
            f"| {r['model']} | {r['index']} | {p['fit']['mse']:.4f} / {p['check']['mse']:.4f} / {p['test']['mse']:.4f} | "
            f"{real['fit']['mse']:.4f} / {real['check']['mse']:.4f} / {real['test']['mse']:.4f} |"
        )
    with open(os.path.join(out_dir, "comparison_placebo.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "comparison_placebo.json"), "w") as f:
        json.dump(records, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="gpt-6.1-sol")
    parser.add_argument("--runs-per-model", type=int, default=3)
    parser.add_argument("--base-dir", default="runs/base_etth1_h96")
    parser.add_argument("--out", default="runs/controls/placebo")
    parser.add_argument("--seed-base", type=int, default=20261106)
    parser.add_argument("--block", type=int, default=192)
    parser.add_argument("--method", choices=["permutation", "bootstrap"], default="permutation")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--max-checks", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=250)
    parser.add_argument("--api-timeout", type=float, default=600.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--context-chars", type=int, default=250000)
    args = parser.parse_args()

    os.chdir(REPO)
    os.makedirs(args.out, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in SHORT]
    if unknown:
        raise SystemExit(f"unknown model short names: {unknown}")
    records = []
    for model in models:
        for index in range(1, args.runs_per_model + 1):
            record = run_one(model, index, args)
            records.append(record)
            write_comparison(records, args.out)
            print(f"{model} run {index}: placebo test {fmt(record['placebo']['test'])}, real test {fmt(record['real_test'])}")
    print(f"written to {args.out}/comparison_placebo.md")


if __name__ == "__main__":
    main()
