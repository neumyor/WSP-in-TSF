"""LLM loop that turns model tool calls into session actions."""

import json
import os
import time

from .session import KernelError
from .tools import TOOLS


FINAL_TURN_NOTICE = "This is your final turn. Call submit now."
BUDGET_NOTICE = (
    "The forward budget is exhausted. You can still run Python without forward queries, "
    "use your remaining checks, and you must call submit."
)
IDLE_NOTICE = "Continue with a tool call: python, check, or submit."


class Driver:
    def __init__(
        self,
        client,
        session,
        run_dir,
        model,
        system_prompt,
        user_prompt,
        max_turns,
        temperature,
        max_tokens,
        context_chars,
    ):
        self.client = client
        self.session = session
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_chars = context_chars
        self.transcript_path = os.path.join(run_dir, "messages.jsonl")
        self.meta_path = os.path.join(run_dir, "driver_meta.json")
        self.usage = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        self.turns = 0

    def _log(self, message, **extra):
        with open(self.transcript_path, "a") as f:
            f.write(json.dumps(dict(message, **extra), ensure_ascii=False) + "\n")

    def _append(self, messages, message, **extra):
        messages.append(message)
        self._log(message, **extra)

    def _call_api(self, messages):
        last_error = None
        for attempt in range(3):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as exc:
                last_error = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"model API failed after retries: {last_error}")

    def _trim(self, messages):
        def size(items):
            return len(json.dumps(items, ensure_ascii=False))

        if size(messages) <= self.context_chars:
            return messages
        head = messages[:2] + [
            {
                "role": "user",
                "content": (
                    "Earlier messages were removed to fit the context window. The sandbox state, "
                    "your variables, and all candidates persist. Use wsp.info() to re-orient."
                ),
            }
        ]
        tail = messages[2:]
        while tail and (tail[0].get("role") == "tool" or size(head + tail) > self.context_chars):
            tail.pop(0)
        return head + tail

    def _execute(self, name, arguments):
        if name == "python":
            return self.session.python(arguments.get("code", ""))
        if name == "check":
            return self.session.check(arguments.get("note", ""))
        if name == "submit":
            return self.session.submit(arguments.get("reason", ""), arguments.get("candidate") or None)
        return dict(error=f"unknown tool {name!r}")

    def run(self):
        messages = []
        self._append(messages, {"role": "system", "content": self.system_prompt})
        self._append(messages, {"role": "user", "content": self.user_prompt})
        budget_notice_sent = False
        idle_turns = 0
        stop_reason = None
        try:
            while not self.session.submitted:
                if self.turns >= self.max_turns:
                    stop_reason = "turn limit reached"
                    break
                if not self.session.kernel_alive:
                    stop_reason = "sandbox kernel stopped"
                    break
                self.turns += 1
                if self.turns == self.max_turns:
                    self._append(messages, {"role": "user", "content": FINAL_TURN_NOTICE})
                messages = self._trim(messages)
                response = self._call_api(messages)
                usage = response.usage
                if usage is not None:
                    self.usage["prompt_tokens"] += int(usage.prompt_tokens or 0)
                    self.usage["completion_tokens"] += int(usage.completion_tokens or 0)
                    self.usage["total_tokens"] += int(usage.total_tokens or 0)
                if not response.choices:
                    idle_turns += 1
                    if idle_turns >= 3:
                        stop_reason = "three turns without a tool call"
                        break
                    continue
                choice = response.choices[0]
                message = choice.message

                assistant = {"role": "assistant", "content": message.content or ""}
                reasoning = getattr(message, "reasoning_content", None) or (
                    getattr(message, "model_extra", None) or {}
                ).get("reasoning_content")
                if reasoning:
                    assistant["reasoning_content"] = reasoning
                tool_calls = message.tool_calls or []
                if tool_calls:
                    assistant["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.function.name, "arguments": call.function.arguments},
                        }
                        for call in tool_calls
                    ]
                self._append(
                    messages,
                    assistant,
                    turn=self.turns,
                    finish_reason=choice.finish_reason,
                    usage=None if usage is None else dict(prompt=usage.prompt_tokens, completion=usage.completion_tokens),
                )

                if not tool_calls:
                    idle_turns += 1
                    if idle_turns >= 3:
                        stop_reason = "three turns without a tool call"
                        break
                    self._append(messages, {"role": "user", "content": IDLE_NOTICE})
                    continue
                idle_turns = 0

                for call in tool_calls:
                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                        result = self._execute(call.function.name, arguments)
                    except json.JSONDecodeError as exc:
                        result = dict(error=f"tool arguments are not valid JSON: {exc}")
                    except KernelError as exc:
                        result = dict(error=str(exc))
                    except Exception as exc:
                        result = dict(error=f"{type(exc).__name__}: {exc}")
                    content = json.dumps(result, ensure_ascii=False, default=str)
                    self._append(messages, {"role": "tool", "tool_call_id": call.id, "content": content})
                    if self.session.submitted:
                        stop_reason = "agent submitted"
                        break

                if self.session.forward_remaining <= 0 and not budget_notice_sent and not self.session.submitted:
                    self._append(messages, {"role": "user", "content": BUDGET_NOTICE})
                    budget_notice_sent = True
        except Exception as exc:
            stop_reason = f"driver error: {type(exc).__name__}: {exc}"

        if not self.session.submitted:
            self.session.submit(f"driver: {stop_reason}", submitted_by="driver")
        self.session.close()
        meta = dict(
            model=self.model,
            turns=self.turns,
            stop_reason=stop_reason,
            usage=self.usage,
            forward_used=self.session.forward_used,
            checks_used=self.session.selector.checks_used,
        )
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return meta
