"""Agent-facing `wsp` client; every call is a request to the driver process."""


class WSPApi:
    """Calls for reading, editing, and running the network. The network itself lives outside this sandbox."""

    def __init__(self, request):
        self._request = request

    def info(self):
        """Window and output sizes, the number of queryable windows, and the remaining forward budget."""
        return self._request("info")

    def list_modules(self):
        """Names, types, and parameter counts of the network's modules."""
        return self._request("list_modules")

    def list_tensors(self):
        """Name, shape, and summary statistics of every weight tensor."""
        return self._request("list_tensors")

    def get_tensor(self, name):
        """A copy of one weight tensor as a numpy array."""
        return self._request("get_tensor", name)

    def set_tensor(self, name, value):
        """Overwrite one weight tensor with finite values of the same shape."""
        return self._request("set_tensor", name, value)

    def forward(self, indices, capture=None):
        """Run the network on queryable windows; costs one forward query.

        indices: integer window indices, at most `max_windows_per_forward` of them.
        capture: module names whose inputs and outputs to record.
        Returns a dict of numpy arrays: indices, input, target, pred, revin_mean, revin_std,
        mse, mae, and captures[module] as a list of {"args", "kwargs", "output"} per call.
        """
        return self._request("forward", indices, capture=capture)
