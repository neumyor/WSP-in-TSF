# phaseformer-wsp requirements

## Goal

Train a phase-only PhaseFormer checkpoint on ETTh1 and let DeepSeek-V4.1-Flash edit its weights directly to reduce forecasting error. The agent sees only the earlier part of the validation split, selects candidates on a later part through a low-information channel, and submits one edited weight vector. The test split is scored once after submission.

## Agent prompt

- The system prompt explains the idea of weight-space programming, states the task, and lists the interface with its budgets.
- Model source is mounted read-only at `model_source/` (configuration at `config/`) for reference. The agent can read it and cannot change it.
- The prompt directs the agent to measure how strongly weights affect the output and to edit the paths that carry signal.
- It contains no description of the model architecture or the data; the agent learns both from the sandbox.
- It places no restriction on how weights are computed. Fitting weights to data is allowed; the prompt asks the agent to understand what each change does to the network and why it should hold on unseen data.
- It asks the agent to use up as much of the check budget as it can before submitting.

## Fixed experimental choices

- Dataset: ETTh1, hourly, seven variables, standard 12/4/4-month split.
- Lookback 720, horizon 96.
- Base model: phase-only PhaseFormer, 1,144 parameters, channel-independent, RevIN instance normalization.
- Base training: train split, checkpoint selected by validation loss, seed 2021.
- Driver: `gpt-6.1-sol` through the OpenAI-compatible endpoint.

## Validation split

- Validation windows are split in time into a fit segment and a check segment.
- The check segment starts `lookback + horizon` windows after the last fit window, so no fit window and check window share a time step.
- Default sizes: half of the remaining windows each.

## Agent interface

- A single python call may run for up to 60 minutes. Variables and files in the work directory persist across calls.
- `wsp.forward(indices, capture)` costs one forward query. It accepts up to 512 fit-segment windows and returns inputs, targets, forecasts, RevIN statistics, and the inputs and outputs of the requested modules as numpy arrays stored in the sandbox. Nothing enters the conversation unless the agent prints it.
- Default forward budget: 100 queries.
- Tensor calls are free: list modules, list tensors, read a tensor, write a tensor with unchanged shape and finite values. `help(wsp)` documents every call.
- The `check` tool evaluates the current weights on the whole check segment and returns one scalar MSE. Default limit: 10 calls. Each checked state becomes a candidate.
- The `submit` tool submits one checked candidate, by default the one with the lowest check MSE. Candidate `c0` is the unedited checkpoint. The submission carries the agent's diagnosis, the edited tensors and entries, and the expected effect.
- The network, its weights, and the fit-segment windows live in the driver process. Agent code runs in a separate process under macOS `sandbox-exec` that can read only the Python runtime and its own work directory, has no network access, and reaches the network only through `wsp` calls. Agent code cannot read the model source, the dataset, check-segment data, test data, or the evaluator.
- When the agent does not submit (turn limit, idle turns, API failure, kernel failure), the driver submits the best checked candidate and records why.

## Gradient baseline

- A gradient baseline is trained on the fit segment with Adam over two parameter scopes (all parameters, output head) and three learning rates, with early stopping on the check segment. It sees the same data as the agent and the same selection channel.
- The report places the agent next to this baseline: both gains over the unedited checkpoint, and the paired difference between agent and baseline on the check segment and the test split, with 95% moving-block bootstrap intervals (block length 2 × horizon). An interval is reported as unavailable when a segment holds fewer than two blocks.
- The agent runs independently of the baseline result.

## Sandbox boundaries

- Weights can be overwritten but not added, removed, or reshaped.
- Any computation is allowed inside the sandbox, including autograd on arrays returned by `wsp`.

## Deliverables

- Base training and artifact export.
- Gradient baseline.
- Sandboxed agent run with forward ledger, check ledger, transcript, and submission.
- Report with base and submitted MSE/MAE on the fit segment, check segment, and test split, paired gains with intervals, the comparison with the gradient baseline, edit footprint, and token use.

## Repository isolation

The local and server copies live in new directories named `phaseformer-wsp`. The existing `PhaseFormer` repositories are read-only sources and are never modified.
