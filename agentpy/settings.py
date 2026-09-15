"""Plumbing for declarative configs; users edit config.py."""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from .interfaces import AgentConfig, LLMSpec, PipeContext, Spawn

@dataclass
class MemoryEntry:
    kind: str
    content: str
    evidence: str = ""
    salience: float = 0.5
    tags: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def render(self):
        return f"- [{self.kind}] {self.content}"

@dataclass(frozen=True)
class Prompt:
    """Only author-written placeholders are substituted; no extra instructions."""
    text: str

    def __call__(self, ctx):
        values = dict(ctx.extras)
        if "context" not in values:
            values["context"] = ctx.agent.inject()
        values["task"] = ctx.payload
        return self.text.format_map(values).strip()

class FlowConfig(AgentConfig):
    memory_schema = MemoryEntry
    spec = LLMSpec()
    flow = ()
    ask = staticmethod(input)
    respond = staticmethod(print)

    def default_spec(self):
        return self.spec

    def pipeline(self, agent):
        return self.flow

    def build_prompt(self, ctx):
        return self.default_prompt(ctx)

    def consolidation_prompt(self, agent):
        return self.memory_prompt(PipeContext(agent=agent))

    def parse_memory(self, raw):
        rows = json.loads(raw)
        if not isinstance(rows, list):
            raise ValueError("Memory output must be a JSON list")
        entries = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"kind", "content", "evidence", "salience", "tags"}:
                raise ValueError("Invalid memory fields")
            if not isinstance(row["kind"], str) or row["kind"] not in {"fact", "preference", "decision", "open_loop", "skill"}:
                raise ValueError("Invalid memory kind")
            if not isinstance(row["content"], str) or not row["content"].strip() or not isinstance(row["evidence"], str):
                raise ValueError("Memory content and evidence must be strings")
            score = row["salience"]
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Salience must be between 0 and 1")
            if not isinstance(row["tags"], list) or not all(isinstance(tag, str) for tag in row["tags"]):
                raise ValueError("Tags must be strings")
            entries.append(self.memory_schema(**row))
        return entries

    def consolidate(self, agent):
        ctx = PipeContext(agent=agent)
        Spawn("consolidator", LLMSpec(role="memory"), self.memory_prompt).run(ctx)
        merge_memory(ctx)

    def ask_user(self, agent, question):
        return self.ask(question)

    def respond_to_user(self, agent, message):
        self.respond(message)

def merge_memory(ctx):
    entries = ctx.agent.config.parse_memory(ctx.last_output)
    merged = {(row.kind, row.content): row for row in ctx.agent.memory}
    merged.update({(row.kind, row.content): row for row in entries})
    ctx.agent.memory.replace_all(list(merged.values()))
    ctx.extras["memory_entries"] = entries

def progress(message):
    """Activity only; full transcripts remain available in agent.sessions."""
    if message.startswith("🧠"):
        print("Working…", flush=True)
    elif message.startswith("✅"):
        print("Done.", flush=True)
    elif message.startswith("❌"):
        print(message, flush=True)
