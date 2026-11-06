TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": (
                "Execute Python in the persistent sandbox. Variables persist across calls. "
                "Returns stdout, stderr, any error, and the forward budget."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source to execute."}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check",
            "description": (
                "Score the current weights on later data that you cannot query. Returns one MSE and a "
                "candidate id for these weights."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "What the current weights change and why."}
                },
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "End the episode by submitting one checked candidate. In the reason, explain what was wrong "
                "in the network, which weights you changed, and why the change should hold on unseen data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "candidate": {
                        "type": "string",
                        "description": "Candidate id from a check result. Defaults to the lowest check MSE.",
                    },
                },
                "required": ["reason"],
            },
        },
    },
]
