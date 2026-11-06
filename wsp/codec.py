"""JSON encoding of nested values that contain numpy arrays, without pickle."""

import base64
import io

import numpy as np


ARRAY_KEY = "__ndarray__"


def encode(value):
    if isinstance(value, np.ndarray):
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        return {ARRAY_KEY: base64.b64encode(buffer.getvalue()).decode("ascii")}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def decode(value):
    if isinstance(value, dict):
        if set(value) == {ARRAY_KEY}:
            return np.load(io.BytesIO(base64.b64decode(value[ARRAY_KEY])), allow_pickle=False)
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value
