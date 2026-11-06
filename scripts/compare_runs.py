"""Collect completed WSP runs into one comparison table."""

import argparse
import glob
import json
import os


def load_run(run_dir):
    report_path = os.path.join(run_dir, "final_report.json")
    meta_path = os.path.join(run_dir, "driver_meta.json")
    if not (os.path.exists(report_path) and os.path.exists(meta_path)):
        return None
    with open(report_path) as f:
        report = json.load(f)
    with open(meta_path) as f:
        meta = json.load(f)
    check = report["metrics"]["check"]
    test = report["metrics"]["test"]
    return dict(
        run=os.path.basename(run_dir),
        model=meta["model"],
        turns=meta["turns"],
        forwards=meta["forward_used"],
        checks=meta["checks_used"],
        tokens=meta["usage"]["total_tokens"],
        candidate=report["submission"]["candidate"],
        check_mse=check["submitted"]["mse"],
        check_gain=check["gain"],
        test_mse=test["submitted"]["mse"],
        test_gain=test["gain"],
        changed=report["footprint"]["changed"],
        tensors=[row["name"] for row in report["footprint"]["tensors"]],
        stop=meta["stop_reason"],
    )


def fmt_gain(gain):
    if gain["ci"] is None:
        return f"{gain['gain']:+.6f}"
    return f"{gain['gain']:+.6f} [{gain['ci'][0]:+.6f}, {gain['ci'][1]:+.6f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs/wsp_h96_*")
    parser.add_argument("--out", default="runs/comparison")
    args = parser.parse_args()

    paths = []
    for pattern in args.runs.split(","):
        paths.extend(glob.glob(pattern.strip()))
    rows = [row for path in sorted(set(paths)) if (row := load_run(path))]
    rows.sort(key=lambda row: (row["model"], row["run"]))
    if not rows:
        raise SystemExit("no completed runs found")

    lines = [
        "# WSP run comparison",
        "",
        "| model | run | turns | fwd | checks | tokens | candidate | check MSE | check gain [95% CI] | test MSE | test gain [95% CI] | changed |",
        "|---|---|---:|---:|---:|---:|---|---:|---|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['run']} | {row['turns']} | {row['forwards']} | {row['checks']} | "
            f"{row['tokens']} | {row['candidate']} | {row['check_mse']:.6f} | {fmt_gain(row['check_gain'])} | "
            f"{row['test_mse']:.6f} | {fmt_gain(row['test_gain'])} | {row['changed']} |"
        )

    lines += ["", "## Per-model summary", "", "| model | runs | mean test gain | best test gain | runs with test CI above zero | edits |", "|---|---:|---:|---:|---:|---|"]
    for model in sorted({row["model"] for row in rows}):
        group = [row for row in rows if row["model"] == model]
        gains = [row["test_gain"]["gain"] for row in group]
        significant = sum(1 for row in group if row["test_gain"]["ci"] and row["test_gain"]["ci"][0] > 0)
        edits = sorted({"/".join(row["tensors"]) or "none" for row in group})
        lines.append(
            f"| {model} | {len(group)} | {sum(gains) / len(gains):+.6f} | {max(gains):+.6f} | {significant} | {'; '.join(edits)} |"
        )

    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(args.out + ".json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\n".join(lines))
    print(f"\nwritten to {args.out}.md")


if __name__ == "__main__":
    main()
