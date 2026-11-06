"""Post-submission evaluation on the fit segment, the check segment, and the locked test split."""

import json
import os

from .model import load_state, mean_losses, score_state
from .stats import paired_gain


FOOTPRINT_THRESHOLD = 1e-4
COMPARED_SEGMENTS = ("check", "test")


def edit_footprint(base_state, edited_state):
    rows = []
    for name, base in base_state.items():
        delta = edited_state[name].float() - base.float()
        changed = int((delta.abs() > FOOTPRINT_THRESHOLD).sum())
        if changed:
            rows.append(
                dict(name=name, changed=changed, numel=int(base.numel()), l2=float(delta.norm()), max_abs=float(delta.abs().max()))
            )
    rows.sort(key=lambda row: row["changed"], reverse=True)
    return dict(
        changed=sum(row["changed"] for row in rows),
        total=int(sum(v.numel() for v in base_state.values())),
        tensors=rows,
    )


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_run(model, base_dir, run_dir, segments, baseline, block):
    base_state = load_state(os.path.join(base_dir, "base_state.pt"))
    submitted_state = load_state(os.path.join(run_dir, "submission.pt"))
    baseline_state = load_state(baseline["state_path"]) if baseline else None
    metrics = {}
    for name, (x, y) in segments.items():
        base = score_state(model, base_state, x, y)
        submitted = score_state(model, submitted_state, x, y)
        row = dict(
            base=mean_losses(*base),
            submitted=mean_losses(*submitted),
            gain=paired_gain(base[0], submitted[0], block),
        )
        if baseline_state is not None and name in COMPARED_SEGMENTS:
            tuned = score_state(model, baseline_state, x, y)
            row["gradient_baseline"] = mean_losses(*tuned)
            row["gradient_baseline_gain"] = paired_gain(base[0], tuned[0], block)
            row["agent_minus_baseline"] = paired_gain(tuned[0], submitted[0], block)
        metrics[name] = row

    forwards = _read_jsonl(os.path.join(run_dir, "forwards.jsonl"))
    with open(os.path.join(run_dir, "submission.json")) as f:
        submission = json.load(f)
    with open(os.path.join(run_dir, "driver_meta.json")) as f:
        driver = json.load(f)
    report = dict(
        driver=driver,
        submission=submission,
        metrics=metrics,
        gradient_baseline=None if baseline is None else {k: baseline["selected"][k] for k in ("scope", "lr", "step")},
        footprint=edit_footprint(base_state, submitted_state),
        forward_queries=len(forwards),
        distinct_fit_windows=len({i for row in forwards for i in row["indices"]}),
        checks=_read_jsonl(os.path.join(run_dir, "checks.jsonl")),
    )
    with open(os.path.join(run_dir, "final_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(run_dir, "report.md"), "w") as f:
        f.write(render_markdown(report))
    return report


def format_gain(gain):
    if gain["ci"] is None:
        return f"{gain['gain']:+.6f} (interval unavailable: segment shorter than two bootstrap blocks)"
    return f"{gain['gain']:+.6f} [{gain['ci'][0]:+.6f}, {gain['ci'][1]:+.6f}]"


def render_markdown(report):
    driver, submission = report["driver"], report["submission"]
    lines = [
        "# WSP run",
        "",
        f"- driver model: {driver['model']}",
        f"- turns: {driver['turns']}, stop: {driver['stop_reason']}",
        f"- tokens: {driver['usage']['total_tokens']} (prompt {driver['usage']['prompt_tokens']}, completion {driver['usage']['completion_tokens']})",
        f"- forward queries: {report['forward_queries']}, distinct fit windows: {report['distinct_fit_windows']}",
        f"- checks: {driver['checks_used']}",
        f"- submitted candidate: {submission['candidate']} by {submission['submitted_by']}",
        "",
        "## Metrics",
        "",
        "Gains are base MSE minus submitted MSE; positive means the submission is better.",
        "",
        "| segment | base MSE | submitted MSE | base MAE | submitted MAE | MSE gain [95% CI] |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, row in report["metrics"].items():
        lines.append(
            f"| {name} | {row['base']['mse']:.6f} | {row['submitted']['mse']:.6f} | {row['base']['mae']:.6f} | "
            f"{row['submitted']['mae']:.6f} | {format_gain(row['gain'])} |"
        )
    if report["gradient_baseline"] is not None:
        selected = report["gradient_baseline"]
        lines += [
            "",
            "## Gradient baseline",
            "",
            f"Adam on the fit segment with early stopping on the check segment; selected scope {selected['scope']}, "
            f"lr {selected['lr']}, step {selected['step']}. Agent minus baseline is baseline MSE minus submitted MSE; "
            "positive means the agent is better.",
            "",
            "| segment | baseline MSE | baseline gain [95% CI] | agent minus baseline [95% CI] |",
            "|---|---:|---|---|",
        ]
        for name in COMPARED_SEGMENTS:
            row = report["metrics"][name]
            lines.append(
                f"| {name} | {row['gradient_baseline']['mse']:.6f} | {format_gain(row['gradient_baseline_gain'])} | "
                f"{format_gain(row['agent_minus_baseline'])} |"
            )
    lines += [
        "",
        "## Checks",
        "",
        "| candidate | check MSE | forward used | note |",
        "|---|---:|---:|---|",
    ]
    for row in report["checks"]:
        note = " ".join(str(row["note"]).replace("|", "/").split())
        lines.append(f"| {row['id']} | {row['check_mse']:.6f} | {row['forward_used']} | {note} |")
    footprint = report["footprint"]
    lines += ["", "## Edit footprint", "", f"- changed entries (|delta| > {FOOTPRINT_THRESHOLD:g}): {footprint['changed']} / {footprint['total']}"]
    for row in footprint["tensors"]:
        lines.append(f"- {row['name']}: {row['changed']} / {row['numel']}, L2 {row['l2']:.6f}, max |delta| {row['max_abs']:.6f}")
    lines += ["", "## Submission reason", "", submission["reason"], ""]
    return "\n".join(lines)
