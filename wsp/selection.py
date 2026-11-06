"""Parent-side check segment: scores candidate weights with a scalar MSE and tracks the best one."""

import json
import os
import shutil

import numpy as np

from .model import load_state, score_state


class CheckSelector:
    def __init__(self, model, x_check, y_check, base_state_path, candidates_dir, ledger_path, max_checks):
        self.model = model
        self.x_check = x_check
        self.y_check = y_check
        self.candidates_dir = candidates_dir
        self.ledger_path = ledger_path
        self.max_checks = int(max_checks)
        os.makedirs(candidates_dir, exist_ok=True)
        base_mse = float(score_state(model, load_state(base_state_path), x_check, y_check)[0].mean())
        self.candidates = [
            dict(id="c0", path=base_state_path, check_mse=base_mse, note="unedited checkpoint", forward_used=0)
        ]
        self.best = self.candidates[0]

    @property
    def base_check_mse(self):
        return self.candidates[0]["check_mse"]

    @property
    def checks_used(self):
        return len(self.candidates) - 1

    @property
    def checks_remaining(self):
        return self.max_checks - self.checks_used

    def check(self, dumped_path, note, forward_used):
        if self.checks_remaining <= 0:
            raise RuntimeError(f"check limit reached ({self.max_checks}); submit a checked candidate")
        candidate_id = f"c{len(self.candidates)}"
        path = os.path.join(self.candidates_dir, f"{candidate_id}.pt")
        shutil.move(dumped_path, path)
        mse = float(score_state(self.model, load_state(path), self.x_check, self.y_check)[0].mean())
        record = dict(
            id=candidate_id,
            path=path,
            check_mse=mse,
            note=note,
            forward_used=int(forward_used),
        )
        self.candidates.append(record)
        if np.isfinite(mse) and mse < self.best["check_mse"]:
            self.best = record
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return dict(
            candidate=record["id"],
            check_mse=mse,
            base_check_mse=self.base_check_mse,
            best_candidate=self.best["id"],
            best_check_mse=self.best["check_mse"],
            checks_remaining=self.checks_remaining,
        )

    def get(self, candidate_id=None):
        if candidate_id is None:
            return self.best
        for record in self.candidates:
            if record["id"] == candidate_id:
                return record
        raise KeyError(f"unknown candidate {candidate_id!r}; checked candidates: {[c['id'] for c in self.candidates]}")
