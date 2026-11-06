# WSP-in-TSF

Weight-space programming on a 1,144-parameter PhaseFormer checkpoint for ETTh1. An LLM agent edits the trained weights through a sandboxed Python interface. A driver process holds the model, scores candidates on a later validation segment, and evaluates the submitted weights once on a locked test split.

## Architecture

```
agent process (macOS sandbox-exec)            driver process
┌────────────────────────────────┐   JSON    ┌──────────────────────────────────┐
│ Python code                     │   RPC     │ Network: PhaseFormer + weights   │
│ wsp.forward / get / set         │ ────────> │ fit windows, forward budget      │
│ model_source/ read-only         │ <──────── │ check scorer, test evaluator     │
└────────────────────────────────┘           └──────────────────────────────────┘
```

- The driver owns the real model, the fit-segment windows, the check segment, and the test split.
- Agent code runs in a separate process under `sandbox-exec`. It reads the model source and `config/` read-only, and reaches the network only through `wsp` calls.
- `wsp.forward` returns predictions, targets, RevIN statistics, and captured module inputs/outputs as numpy arrays. Each call costs one forward query.
- `check` scores the current weights on a later validation segment that the agent never sees, and returns one scalar MSE. Only checked candidates can be submitted.
- `submit` picks a candidate. The driver evaluates it once on the test split and writes the report.

Validation windows are split in time into a fit segment and a check segment separated by `lookback + horizon` windows, so no fit and check window share a time step.

## Quick start

Requirements: Python 3.10+, `uv`, and macOS for the sandboxed agent. Base training also runs on Linux.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch numpy pandas scikit-learn easydict pytorch-lightning openai einops
cp .env.example .env   # set WSP_API_KEY
```

Place `ETTh1.csv` at `resources/all_datasets/ETT/ETTh1.csv`.

Train a base checkpoint:

```bash
.venv/bin/python scripts/train_base.py --horizon 96 --out runs/base_etth1_h96
```

Measure the gradient baseline:

```bash
.venv/bin/python run_wsp.py baseline --base-dir runs/base_etth1_h96
```

Run the agent:

```bash
.venv/bin/python run_wsp.py agent --base-dir runs/base_etth1_h96 --run-dir runs/wsp_h96_run1 --model gpt-6.1-sol
```

Run the scripted smoke episode without an LLM:

```bash
.venv/bin/python run_wsp.py smoke --base-dir runs/base_etth1_h96 --run-dir runs/smoke
```

Collect completed runs:

```bash
.venv/bin/python scripts/compare_runs.py --runs "runs/wsp_h96_*" --out runs/comparison
```

Each run directory contains the transcript, `forwards.jsonl`, `checks.jsonl`, candidate and submission weights, `report.md`, and `final_report.json`.

## Repository layout

- `src/`, `config/`: PhaseFormer model and data code, copied from the local PhaseFormer repository. The sandbox mounts them read-only.
- `wsp/`: driver-side network, sandbox kernel, check scorer, gradient baseline, driver loop, and report.
- `scripts/`: base training, diagnosis, and run comparison.
- `run_wsp.py`: CLI with `baseline`, `agent`, and `smoke`.
- `runs/`, `resources/`, `.env`: gitignored artifacts, data, and secrets.

## Development conventions

- Follow the project behavior norms: no hashes or checksums, no redundant audit layers, no over-design; write final-state documentation; set LLM token budgets generously.
- Keep the existing PhaseFormer repositories untouched. `src/` and `config/` are copies used by this project.
- Check and test data stay in the driver process. Never expose them to the sandbox or the agent.
- Budget accounting belongs to the driver (`wsp/network.py`). Agent code only calls `wsp`.
- Before committing, run `ruff check --select F run_wsp.py wsp scripts` and `run_wsp.py smoke`.
- Never commit `.env`, `runs/`, `resources/`, checkpoints, or exported data.

The fixed protocol, split rules, and budgets are in `REQUIREMENTS.md`.
