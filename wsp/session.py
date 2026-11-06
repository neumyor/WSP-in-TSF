"""One episode: the network, the restricted code executor, the check segment, and the submission."""

import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading

from .codec import decode, encode
from .network import call


KERNEL_FILES = ("__init__.py", "codec.py", "sandbox.py", "kernel.py")


class KernelError(RuntimeError):
    pass


def _profile(workdir, readable):
    allowed = " ".join(f'(subpath "{path}")' for path in [workdir, *readable])
    return f"""(version 1)
(allow default)
(deny network*)
(deny file-read-data (subpath "/Users") (subpath "/Volumes") (subpath "/private/tmp") (subpath "/private/var/folders"))
(allow file-read-data {allowed})
(deny file-write*)
(allow file-write* (subpath "{workdir}") (literal "/dev/null"))
"""


class Kernel:
    """Agent code runs in a separate process that can read only the Python runtime and its own work directory."""

    def __init__(self, repo_root, run_dir, timeout):
        self.repo_root = repo_root
        self.workdir = os.path.realpath(os.path.join(run_dir, "sandbox"))
        self.stderr_path = os.path.join(run_dir, "kernel_stderr.log")
        self.timeout = timeout
        self.proc = None
        self.lines = None
        self.stderr_file = None

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        package = os.path.join(self.workdir, "wsp")
        os.makedirs(os.path.join(self.workdir, "tmp"), exist_ok=True)
        os.makedirs(package, exist_ok=True)
        for name in KERNEL_FILES:
            shutil.copyfile(os.path.join(self.repo_root, "wsp", name), os.path.join(package, name))
        for name, target in (("model_source", "src/models"), ("config", "config")):
            link = os.path.join(self.workdir, name)
            if not os.path.exists(link):
                os.symlink(os.path.join(self.repo_root, target), link)
        readable = sorted(
            {
                os.path.realpath(sys.prefix),
                os.path.realpath(sys.base_prefix),
                os.path.realpath(os.path.join(self.repo_root, "src", "models")),
                os.path.realpath(os.path.join(self.repo_root, "config")),
            }
        )
        env = dict(
            PATH="/usr/bin:/bin",
            HOME=self.workdir,
            TMPDIR=os.path.join(self.workdir, "tmp"),
            PYTHONPATH=self.workdir,
            PYTHONDONTWRITEBYTECODE="1",
        )
        self.stderr_file = open(self.stderr_path, "a")
        self.proc = subprocess.Popen(
            ["/usr/bin/sandbox-exec", "-p", _profile(self.workdir, readable), sys.executable, "-m", "wsp.kernel"],
            cwd=self.workdir,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
            text=True,
            bufsize=1,
        )
        self.lines = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def send(self, message):
        if not self.alive:
            raise KernelError("the sandbox kernel is not running")
        try:
            self.proc.stdin.write(json.dumps(encode(message)) + "\n")
            self.proc.stdin.flush()
        except OSError as exc:
            self.close()
            raise KernelError(f"the sandbox kernel pipe failed: {exc}")

    def receive(self):
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            raise KernelError(f"the sandbox kernel did not answer within {self.timeout} s and was stopped")
        if line is None:
            self.close()
            raise KernelError("the sandbox kernel exited; see kernel_stderr.log")
        return decode(json.loads(line))

    def close(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
        if self.stderr_file is not None:
            self.stderr_file.close()
            self.stderr_file = None


class Session:
    def __init__(self, network, kernel, selector, run_dir):
        self.network = network
        self.kernel = kernel
        self.selector = selector
        self.run_dir = run_dir
        self.submission = None

    @property
    def submitted(self):
        return self.submission is not None

    @property
    def kernel_alive(self):
        return self.kernel.alive

    @property
    def forward_used(self):
        return self.network.used

    @property
    def forward_remaining(self):
        return self.network.budget - self.network.used

    def start(self):
        self.kernel.start()

    def python(self, code):
        self.kernel.send(dict(code=code))
        while True:
            message = self.kernel.receive()
            if message.get("type") == "done":
                message.pop("type")
                message["forward_used"] = self.forward_used
                message["forward_remaining"] = self.forward_remaining
                return message
            try:
                result = call(self.network, message["name"], message.get("args", []), message.get("kwargs", {}))
                reply = dict(result=result)
            except Exception as exc:
                reply = dict(error=f"{type(exc).__name__}: {exc}")
            self.kernel.send(reply)

    def check(self, note):
        path = os.path.join(self.run_dir, "candidate.pt")
        self.network.save_state(path)
        try:
            return self.selector.check(path, note, self.forward_used)
        except RuntimeError as exc:
            return dict(error=str(exc))

    def submit(self, reason, candidate=None, submitted_by="agent"):
        if self.submitted:
            return dict(error="already submitted")
        try:
            record = self.selector.get(candidate)
        except KeyError as exc:
            return dict(error=str(exc))
        if not math.isfinite(record["check_mse"]):
            return dict(error=f"candidate {record['id']} has a non-finite check MSE and cannot be submitted")
        shutil.copyfile(record["path"], os.path.join(self.run_dir, "submission.pt"))
        self.submission = dict(
            candidate=record["id"],
            check_mse=record["check_mse"],
            base_check_mse=self.selector.base_check_mse,
            note=record["note"],
            reason=reason,
            submitted_by=submitted_by,
            forward_used=self.forward_used,
            checks_used=self.selector.checks_used,
        )
        with open(os.path.join(self.run_dir, "submission.json"), "w") as f:
            json.dump(self.submission, f, indent=2)
        return dict(submitted=True, **self.submission)

    def close(self):
        self.kernel.close()
