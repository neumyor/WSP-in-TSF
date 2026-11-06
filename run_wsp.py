"""Weight-space programming on the exported ETTh1 checkpoint: gradient baseline, agent run, and a scripted smoke run."""

import argparse
import json
import os

from wsp.baseline import load_gradient_baseline, run_gradient_baseline
from wsp.data import load_windows, split_summary, split_validation
from wsp.model import load_model
from wsp.prompts import USER_PROMPT, system_prompt
from wsp.report import evaluate_run, format_gain
from wsp.selection import CheckSelector
from wsp.network import Network
from wsp.session import Kernel, Session


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

SMOKE_CODE = """
info = wsp.info()
r = wsp.forward(np.arange(64), capture=["predictor.decoder", "routing_layers.0.interact.router_sender"])
dec = r["captures"]["predictor.decoder"][0]
print("info", info)
print("shapes", r["input"].shape, r["pred"].shape, r["revin_std"].shape, dec["args"][0].shape, dec["output"].shape)
print("sender output", [None if o is None else o.shape for o in r["captures"]["routing_layers.0.interact.router_sender"][0]["output"]])
bias = wsp.get_tensor("predictor.decoder.bias")
wsp.set_tensor("predictor.decoder.bias", bias + 0.01)
r2 = wsp.forward(np.arange(64))
print("mse", r["mse"], r2["mse"], "remaining", r2["forward_remaining"])
"""


GUARD_CODE = """
import os, socket
attempts = {{
    "read test windows": lambda: open({test!r}, "rb"),
    "read dataset csv": lambda: open({csv!r}),
    "read api key": lambda: open({env!r}),
    "read model source": lambda: open({source!r}),
    "read model source via link": lambda: open("model_source/PhaseFormer.py"),
    "list repository": lambda: os.listdir({repo!r}),
    "overwrite base weights": lambda: open({base!r}, "ab"),
    "network": lambda: socket.create_connection(("1.1.1.1", 80), timeout=3),
}}
for name, attempt in attempts.items():
    try:
        attempt()
        print(name, "ALLOWED")
    except OSError:
        print(name, "blocked")
x = torch.randn(4, requires_grad=True)
(x ** 2).sum().backward()
print("autograd inside the sandbox:", x.grad is not None)
import wsp
"""


def load_env(path):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def prompt_facts(args):
    return dict(budget=args.budget, max_checks=args.max_checks, time_limit=int(args.kernel_timeout // 60))


def prepare_run(args):
    base_dir = os.path.abspath(args.base_dir)
    run_dir = os.path.abspath(args.run_dir)
    if os.path.exists(run_dir):
        raise SystemExit(f"run directory already exists: {run_dir}; choose a new --run-dir")
    os.makedirs(run_dir)

    model, meta = load_model(base_dir)
    x_val, y_val = load_windows(base_dir, "val")
    x_test, y_test = load_windows(base_dir, "test")
    split = split_validation(len(x_val), int(meta["seq_len"]), int(meta["horizon"]), args.check_fraction)
    x_fit, y_fit = x_val[split["fit"]], y_val[split["fit"]]
    x_check, y_check = x_val[split["check"]], y_val[split["check"]]

    base_state_path = os.path.join(base_dir, "base_state.pt")
    selector = CheckSelector(
        model,
        x_check,
        y_check,
        base_state_path,
        os.path.join(run_dir, "candidates"),
        os.path.join(run_dir, "checks.jsonl"),
        args.max_checks,
    )
    network = Network(base_dir, x_fit, y_fit, os.path.join(run_dir, "forwards.jsonl"), args.budget, args.max_windows)
    kernel = Kernel(REPO_ROOT, run_dir, args.kernel_timeout)
    session = Session(network, kernel, selector, run_dir)
    segments = dict(fit=(x_fit, y_fit), check=(x_check, y_check), test=(x_test, y_test))
    facts = prompt_facts(args)
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        config = {k: v for k, v in vars(args).items() if k != "func"}
        json.dump(dict(config, split=split_summary(split)), f, indent=2)
    return model, meta, split, session, segments, facts


def cmd_baseline(args):
    scopes = {"all": "", "head": args.head_prefix}
    try:
        result = run_gradient_baseline(
            os.path.abspath(args.base_dir),
            args.check_fraction,
            scopes,
            args.lrs,
            args.steps,
            args.eval_every,
            args.patience,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"gradient baseline cannot run: {exc}")
    selected = result["selected"]
    print(json.dumps(dict(split=result["split"], base=result["base"]), indent=2))
    print(
        f"selected: scope {selected['scope']}, lr {selected['lr']}, step {selected['step']}\n"
        f"check gain {format_gain(selected['check_gain'])}\n"
        f"test gain  {format_gain(selected['test_gain'])}"
    )


def cmd_agent(args):
    from openai import OpenAI

    from wsp.driver import Driver

    load_env(os.path.join(REPO_ROOT, ".env"))
    api_key = os.environ.get("WSP_API_KEY")
    if not api_key:
        raise SystemExit("WSP_API_KEY is not set")
    model, meta, split, session, segments, facts = prepare_run(args)
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.api_timeout, max_retries=0)
    driver = Driver(
        client=client,
        session=session,
        run_dir=os.path.abspath(args.run_dir),
        model=args.model,
        system_prompt=system_prompt(facts),
        user_prompt=USER_PROMPT,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        context_chars=args.context_chars,
    )
    session.start()
    driver.run()
    finish(model, meta, args, segments, split)


def cmd_smoke(args):
    model, meta, split, session, segments, facts = prepare_run(args)
    system_prompt(facts)
    session.start()
    try:
        response = session.python(SMOKE_CODE)
        print(response["stdout"], response["error"] or "")
        base_dir = os.path.abspath(args.base_dir)
        guard = GUARD_CODE.format(
            test=os.path.join(base_dir, "test_data.pt"),
            csv=os.path.join(REPO_ROOT, "resources", "all_datasets", "ETT", "ETTh1.csv"),
            env=os.path.join(REPO_ROOT, ".env"),
            base=os.path.join(base_dir, "base_state.pt"),
            source=os.path.join(REPO_ROOT, "src", "models", "PhaseFormer.py"),
            repo=REPO_ROOT,
        )
        print(session.python(guard)["stdout"])
        print("wsp after rebinding:", session.python("print(type(wsp).__name__)")["stdout"].strip())
        print("check", session.check("decoder bias + 0.01"))
        print("submit", session.submit("smoke run", candidate="c1"))
    finally:
        session.close()
    with open(os.path.join(os.path.abspath(args.run_dir), "driver_meta.json"), "w") as f:
        json.dump(
            dict(model="scripted", turns=0, stop_reason="smoke", usage=dict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                 forward_used=session.forward_used, checks_used=session.selector.checks_used),
            f,
            indent=2,
        )
    finish(model, meta, args, segments, split)


def finish(model, meta, args, segments, split):
    base_dir = os.path.abspath(args.base_dir)
    baseline = load_gradient_baseline(base_dir, split)
    if baseline is None:
        print("no gradient baseline with this split; run `run_wsp.py baseline` to add the comparison")
    report = evaluate_run(model, base_dir, os.path.abspath(args.run_dir), segments, baseline, block=2 * int(meta["horizon"]))
    for name, row in report["metrics"].items():
        line = f"{name:>5}: base {row['base']['mse']:.6f} submitted {row['submitted']['mse']:.6f} gain {format_gain(row['gain'])}"
        if "agent_minus_baseline" in row:
            line += f" | agent minus gradient baseline {format_gain(row['agent_minus_baseline'])}"
        print(line)
    print(f"report: {os.path.join(os.path.abspath(args.run_dir), 'report.md')}")


def add_common(parser):
    parser.add_argument("--base-dir", default="runs/base_etth1_h96")
    parser.add_argument("--check-fraction", type=float, default=0.5)


def add_run(parser):
    add_common(parser)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--max-windows", type=int, default=512)
    parser.add_argument("--max-checks", type=int, default=10)
    parser.add_argument("--kernel-timeout", type=float, default=3600.0)
    parser.add_argument("--api-timeout", type=float, default=600.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("baseline", help="gradient fine-tuning on the fit segment, reported next to the agent")
    add_common(p)
    p.add_argument("--head-prefix", default="predictor.")
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--patience", type=int, default=10)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("agent", help="LLM-driven episode")
    add_run(p)
    p.add_argument("--max-turns", type=int, default=250)
    p.add_argument("--model", default="gpt-6.1-sol")
    p.add_argument("--base-url", default="https://llm-center.modelbest.co/v1")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--context-chars", type=int, default=250000)
    p.set_defaults(func=cmd_agent)

    p = sub.add_parser("smoke", help="scripted episode that exercises the sandbox, check, submit, and report")
    add_run(p)
    p.set_defaults(func=cmd_smoke)

    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    args.func(args)


if __name__ == "__main__":
    main()
