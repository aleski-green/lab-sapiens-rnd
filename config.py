"""Personality layer.

Edit this file to change memory shape, prompts, pipes, and user IO.
AgentPy itself stays generic — it only injects context and runs the pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

from agentpy.interfaces import (
    Agent,
    AgentConfig,
    Iterate,
    LLMSpec,
    PipeContext,
    PipeFn,
    Spawn,
    Step,
)


# ---------------------------------------------------------------------------
# Memory schema — one row in agent.memory
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    kind: str  # fact | preference | decision | open_loop | skill
    content: str
    evidence: str = ""
    salience: float = 0.5
    tags: list[str] = field(default_factory=list)
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def render(self) -> str:
        tag = ",".join(self.tags) if self.tags else "-"
        return f"- [{self.kind} | {self.salience:.2f} | {tag}] {self.content}"


class Config(AgentConfig):
    memory_schema = MemoryEntry

    # -- specs / prompts ----------------------------------------------------

    def default_spec(self) -> LLMSpec:
        return LLMSpec(model="default", role="worker")

    def build_prompt(self, ctx: PipeContext) -> str:
        return (
            f"{ctx.agent.inject(session_key=ctx.session_key)}\n\n"
            f"# Task\n{ctx.payload}"
        )

    def think_prompt(self, ctx: PipeContext) -> str:
        return (
            f"{ctx.agent.inject()}\n\n"
            "# Task\n"
            "Think about the user message. If you need a clarifying question, "
            "end with a line `ASK: <question>`. Otherwise end with `ASK: none`.\n\n"
            f"{ctx.payload}"
        )

    def act_prompt(self, ctx: PipeContext) -> str:
        extra = ctx.extras.get("user_answer")
        followup = f"\n\n# User answer\n{extra}" if extra else ""
        return (
            f"{ctx.agent.inject(session_key='think')}\n\n"
            "# Task\n"
            "Produce the reply the user should see. No hidden scratchpad."
            f"{followup}"
        )

    def consolidation_prompt(self, agent: Agent) -> str:
        return (
            f"{agent.inject()}\n\n"
            "# Task\n"
            "Rewrite durable memory as a JSON list of objects with keys:\n"
            "kind, content, evidence, salience, tags.\n"
            "kind ∈ fact | preference | decision | open_loop | skill.\n"
            "Keep only durable items. Drop chatter. Return JSON only."
        )

    def parse_memory(self, raw: str) -> list[MemoryEntry]:
        match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
        blob = match.group(0) if match else raw
        rows = json.loads(blob)
        entries = []
        for row in rows:
            entries.append(
                MemoryEntry(
                    kind=row["kind"],
                    content=row["content"],
                    evidence=row.get("evidence", ""),
                    salience=float(row.get("salience", 0.5)),
                    tags=list(row.get("tags", [])),
                )
            )
        return entries

    def consolidate(self, agent: Agent) -> None:
        ctx = PipeContext(agent=agent)
        Spawn(
            "consolidator",
            spec=LLMSpec(role="memory"),
            prompt=lambda c: self.consolidation_prompt(c.agent),
        ).run(ctx)
        try:
            agent.memory.replace_all(self.parse_memory(ctx.last_output or "[]"))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Worker didn't return valid JSON (e.g. EchoFactory). Keep memory.
            return

    # -- user IO ------------------------------------------------------------

    def ask_user(self, agent: Agent, question: str) -> str:
        return input(f"\n[agent {agent.agid}] {question}\n> ")

    def respond_to_user(self, agent: Agent, message: str) -> None:
        print(f"\n[agent {agent.agid}]\n{message}\n")

    # -- lambdas in the pipe ------------------------------------------------

    def gate_ask(self, ctx: PipeContext) -> PipeContext:
        output = ctx.last_output or ""
        ask_line = ""
        for line in reversed(output.splitlines()):
            if line.startswith("ASK:"):
                ask_line = line[4:].strip()
                break
        if ask_line and ask_line.lower() != "none":
            ctx.extras["user_answer"] = ctx.agent.ask_user(ask_line)
        return ctx

    def finish(self, ctx: PipeContext) -> PipeContext:
        if ctx.last_output:
            ctx.agent.respond_to_user(ctx.last_output)
        ctx.agent.consolidate()
        return ctx

    def pipeline(self, agent: Agent) -> Sequence[Step | PipeFn]:
        return [
            Spawn("think", prompt=self.think_prompt),
            self.gate_ask,
            Iterate("think", prompt=self.act_prompt),
            self.finish,
        ]
