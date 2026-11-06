"""Agent code executor: runs in a restricted process and reaches the network only through requests to the driver."""

import contextlib
import io
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
import torch

from wsp.codec import decode, encode
from wsp.sandbox import WSPApi


OUTPUT_LIMIT = 30000


class DriverError(Exception):
    pass


def _truncate(text):
    if len(text) <= OUTPUT_LIMIT:
        return text
    return text[:OUTPUT_LIMIT] + f"\n...[{len(text) - OUTPUT_LIMIT} more characters not shown]"


def main():
    channel_in = os.fdopen(os.dup(0), "r")
    channel_out = os.fdopen(os.dup(1), "w")
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.dup2(2, 1)

    def send(message):
        channel_out.write(json.dumps(encode(message)) + "\n")
        channel_out.flush()

    def receive():
        line = channel_in.readline()
        if not line:
            sys.exit(0)
        return decode(json.loads(line))

    def request(name, *args, **kwargs):
        send(dict(type="call", name=name, args=list(args), kwargs=kwargs))
        reply = receive()
        if reply.get("error"):
            raise DriverError(reply["error"])
        return reply["result"]

    api = WSPApi(request)
    namespace = {"np": np, "torch": torch, "pd": pd}
    while True:
        message = receive()
        namespace["wsp"] = api
        stdout, stderr = io.StringIO(), io.StringIO()
        error = None
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(message["code"], "<wsp>", "exec"), namespace)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}\n" + traceback.format_exc(limit=6)
        send(
            dict(
                type="done",
                stdout=_truncate(stdout.getvalue()),
                stderr=_truncate(stderr.getvalue()),
                error=_truncate(error) if error else None,
            )
        )


if __name__ == "__main__":
    main()
