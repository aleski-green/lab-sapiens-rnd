"""Codex CLI implementation of the AgentPy LLM contracts.

Codex runs locally without sandboxing, streams JSONL lifecycle events to the
caller, and keeps the Codex thread ID so ``LLMFactory.iterate`` can resume the
same session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
from threading import Thread
from typing import Any, Callable
from uuid import uuid4

from agentpy.interfaces import LLM, LLMFactory, LLMSpec, SessionLog


EventSink = Callable[[str], None]


def _print_event(message: str) -> None:
    print(message, flush=True)


@dataclass
class CodexLLM(LLM):
    """One resumable local ``codex exec`` thread."""

    spec: LLMSpec
    workdir: Path
    id: str = field(default_factory=lambda: f"pending_{uuid4().hex[:8]}")
    resume: bool = False
    event_sink: EventSink = _print_event

    def complete(self, prompt: str) -> str:
        command = self._command(prompt)
        final_message: str | None = None
        recent_output: list[str] = []
        stderr_output: list[str] = []

        process = subprocess.Popen(
            command,
            cwd=self.workdir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        assert process.stderr is not None

        def drain_stderr() -> None:
            for stderr_line in process.stderr:
                stderr_output.append(stderr_line.rstrip())
                del stderr_output[:-20]

        stderr_thread = Thread(target=drain_stderr, daemon=True)
        stderr_thread.start()

        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            recent_output.append(line)
            recent_output = recent_output[-20:]

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                self.event_sink(f"[codex] {line}")
                continue

            message = self._consume_event(event)
            if message is not None:
                final_message = message

        returncode = process.wait()
        stderr_thread.join()
        if returncode != 0:
            detail = "\n".join([*stderr_output, *recent_output])
            raise RuntimeError(f"codex exec failed (exit {returncode}):\n{detail}")
        if final_message is None:
            raise RuntimeError("Codex completed without an agent message.")
        return final_message

    def _command(self, prompt: str) -> list[str]:
        common = [
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
        ]
        if self.spec.model != "default":
            common.extend(["--model", self.spec.model])

        if self.resume:
            return ["codex", "exec", "resume", *common, self.id, prompt]

        return [
            "codex",
            "exec",
            *common,
            "--cd",
            str(self.workdir),
            "--color",
            "never",
            prompt,
        ]

    def _consume_event(self, event: dict[str, Any]) -> str | None:
        event_type = event.get("type")

        if event_type == "thread.started":
            self.id = event["thread_id"]
            self.event_sink(f"🧵 Codex thread {self.id} started")
        elif event_type == "turn.started":
            self.event_sink("🧠 Codex is working…")
        elif event_type == "turn.completed":
            usage = event.get("usage", {})
            self.event_sink(
                "✅ Codex turn completed "
                f"(input={usage.get('input_tokens', '?')}, "
                f"output={usage.get('output_tokens', '?')}, "
                f"reasoning={usage.get('reasoning_output_tokens', '?')})"
            )
        elif event_type in {"turn.failed", "error"}:
            self.event_sink(f"❌ Codex error: {event.get('message', event)}")
        elif event_type in {"item.started", "item.completed"}:
            return self._consume_item(event.get("item", {}), completed=event_type.endswith("completed"))

        return None

    def _consume_item(self, item: dict[str, Any], *, completed: bool) -> str | None:
        item_type = item.get("type")
        marker = "finished" if completed else "started"

        if item_type == "agent_message" and completed:
            text = item.get("text", "")
            self.event_sink(f"🤖 Codex response\n{text}")
            return text
        if item_type == "reasoning" and completed:
            text = item.get("text") or item.get("summary")
            if text:
                self.event_sink(f"💭 Reasoning summary\n{text}")
        elif item_type == "command_execution":
            command = item.get("command", "")
            self.event_sink(f"🛠️ Command {marker}: {command}")
        elif item_type == "file_change":
            self.event_sink(f"📝 File change {marker}")
        elif item_type == "mcp_tool_call":
            tool = item.get("tool", item.get("name", "unknown"))
            self.event_sink(f"🔌 MCP call {marker}: {tool}")
        elif item_type == "web_search":
            self.event_sink(f"🔎 Web search {marker}")
        elif item_type == "plan" and completed:
            text = item.get("text")
            if text:
                self.event_sink(f"📋 Plan\n{text}")

        return None


@dataclass
class CodexFactory(LLMFactory):
    """Create and resume unrestricted local Codex CLI sessions."""

    workdir: Path = field(default_factory=Path.cwd)
    event_sink: EventSink = _print_event

    def spawn(self, spec: LLMSpec) -> LLM:
        return CodexLLM(
            spec=spec,
            workdir=self.workdir,
            event_sink=self.event_sink,
        )

    def iterate(self, session: SessionLog) -> LLM:
        if session.llm_id.startswith("pending_"):
            raise RuntimeError(f"Session {session.key!r} has no Codex thread ID yet.")
        return CodexLLM(
            spec=session.spec,
            workdir=self.workdir,
            id=session.llm_id,
            resume=True,
            event_sink=self.event_sink,
        )


def run_codex(prompt: str, *, workdir: Path, model: str | None = None) -> str:
    """Backward-compatible one-shot Codex call."""

    llm = CodexLLM(
        spec=LLMSpec(model=model or "default"),
        workdir=workdir,
    )
    return llm.complete(prompt)
