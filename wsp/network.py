"""Driver-side network, fit-segment windows, tensor access, and the budgeted forward query."""

import json

import numpy as np
import torch

from .model import clone_state, forecast, load_model


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, (tuple, list)):
        return [_to_numpy(item) for item in value]
    return None


class Network:
    def __init__(self, base_dir, x_fit, y_fit, ledger_path, budget, max_windows):
        self.model, self.meta = load_model(base_dir)
        self.inputs = x_fit
        self.targets = y_fit
        self.budget = int(budget)
        self.max_windows = int(max_windows)
        self.used = 0
        self.ledger_path = ledger_path
        self.modules = dict(self.model.named_modules())

    def _tensors(self):
        return {name: value for name, value in self.model.state_dict().items() if value.is_floating_point()}

    def info(self):
        return dict(
            lookback=int(self.meta["seq_len"]),
            horizon=int(self.meta["horizon"]),
            variables=int(self.inputs.shape[-1]),
            fit_windows=int(len(self.inputs)),
            max_windows_per_forward=self.max_windows,
            forward_used=self.used,
            forward_remaining=self.budget - self.used,
            forward_total=self.budget,
        )

    def list_modules(self):
        return [
            dict(name=name, type=type(module).__name__, params=sum(p.numel() for p in module.parameters(recurse=False)))
            for name, module in self.modules.items()
            if name
        ]

    def list_tensors(self):
        rows = []
        for name, value in self._tensors().items():
            flat = value.float().reshape(-1)
            rows.append(
                dict(
                    name=name,
                    shape=list(value.shape),
                    numel=int(flat.numel()),
                    mean=float(flat.mean()),
                    std=float(flat.std()) if flat.numel() > 1 else 0.0,
                    min=float(flat.min()),
                    max=float(flat.max()),
                )
            )
        return rows

    def get_tensor(self, name):
        tensors = self._tensors()
        if name not in tensors:
            raise KeyError(f"unknown tensor {name!r}; see wsp.list_tensors()")
        return tensors[name].cpu().numpy().copy()

    def set_tensor(self, name, value):
        tensors = self._tensors()
        if name not in tensors:
            raise KeyError(f"unknown tensor {name!r}; see wsp.list_tensors()")
        target = tensors[name]
        array = np.asarray(value, dtype=np.float64)
        if array.shape != tuple(target.shape):
            raise ValueError(f"shape mismatch for {name}: got {array.shape}, expected {tuple(target.shape)}")
        update = torch.from_numpy(array).to(target.dtype)
        if not torch.isfinite(update).all():
            raise ValueError(f"value for {name} contains NaN or infinity in {target.dtype}")
        with torch.no_grad():
            target.copy_(update)

    def _indices(self, indices):
        idx = np.asarray(indices)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError("indices must be a non-empty one-dimensional sequence of integers")
        if not np.issubdtype(idx.dtype, np.integer):
            raise TypeError(f"indices must be integers, got dtype {idx.dtype}")
        if idx.size > self.max_windows:
            raise ValueError(f"at most {self.max_windows} windows per forward, got {idx.size}")
        if idx.min() < 0 or idx.max() >= len(self.inputs):
            raise IndexError(f"indices must lie in [0, {len(self.inputs)})")
        return idx.astype(np.int64)

    def forward(self, indices, capture=None):
        if self.used >= self.budget:
            raise RuntimeError(f"forward budget exhausted ({self.used}/{self.budget})")
        idx = self._indices(indices)
        capture = [capture] if isinstance(capture, str) else list(capture or [])
        unknown = [name for name in capture if name not in self.modules]
        if unknown:
            raise KeyError(f"unknown modules {unknown}; see wsp.list_modules()")

        records = {name: [] for name in capture}

        def make_hook(name):
            def hook(_module, args, kwargs, output):
                records[name].append(
                    dict(
                        args=_to_numpy(list(args)),
                        kwargs={k: _to_numpy(v) for k, v in kwargs.items() if torch.is_tensor(v)},
                        output=_to_numpy(output),
                    )
                )

            return hook

        x = self.inputs[idx]
        handles = [self.modules[name].register_forward_hook(make_hook(name), with_kwargs=True) for name in capture]
        try:
            with torch.no_grad():
                pred = forecast(self.model, x)
                stats = self.model.revin.normalize(x)[1] if self.model.use_revin else None
        finally:
            for handle in handles:
                handle.remove()

        target = self.targets[idx]
        diff = pred - target
        self.used += 1
        result = dict(
            query=self.used,
            forward_remaining=self.budget - self.used,
            indices=idx,
            input=x.numpy().copy(),
            target=target.numpy().copy(),
            pred=pred.numpy().copy(),
            revin_mean=None if stats is None else stats[0][:, 0, :].numpy().copy(),
            revin_std=None if stats is None else stats[1][:, 0, :].numpy().copy(),
            mse=float((diff**2).mean()),
            mae=float(diff.abs().mean()),
            captures=records,
        )
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(dict(query=self.used, indices=idx.tolist(), capture=capture, mse=result["mse"])) + "\n")
        return result

    def save_state(self, path):
        torch.save(clone_state(self.model.state_dict()), path)


CALLS = ("info", "list_modules", "list_tensors", "get_tensor", "set_tensor", "forward")


def call(network, name, args, kwargs):
    if name not in CALLS:
        raise AttributeError(f"wsp has no call {name!r}")
    return getattr(network, name)(*args, **kwargs)
