"""Editable manifests, prompt lambdas, and memory policy for the debate demo."""

import json
import math

from agentpy.nodes import AdversarialNode
from config import Config, MemoryEntry


class AdversarialConfig(Config):
    manifests = {
        "proposer": "Propose a concise solution. Distinguish user facts, your proposals, and uncertainties. Do not use tools; respond in text only.",
        "critic": "Critique the proposal against the task and evidence. Identify unsupported assumptions and offer improvements. Do not use tools; respond in text only.",
        "arbiter": (
            "Compile the supported, useful parts of the proposal and critique into a JSON list of memory additions. "
            "Use keys kind, content, evidence, salience, tags. "
            "Kinds: fact, preference, decision, open_loop, skill. Salience: number from 0 to 1; tags: list of strings. "
            "Retain explicit durable user preferences; omit facts marked temporary. "
            "Do not present a proposed solution as a user-approved decision. Record unresolved choices as open_loop. "
            "Do not repeat unrelated existing memory. Return JSON only. Do not use tools."
        ),
    }

    def role_prompt(self, ctx, role):
        # All prompt content is explicit here; the node adds no instructions.
        parts = [self.manifests[role], ctx.extras["context"], f"# Task\n{ctx.payload}"]
        if role in ("critic", "arbiter"):
            parts.append(f"# Proposal\n{ctx.extras['proposal']}")
        if role == "arbiter":
            parts.append(f"# Critique\n{ctx.extras['critique']}")
        return "\n\n".join(parts)

    def merge_memory(self, ctx):
        # Validate everything before touching the existing store; errors surface.
        rows = json.loads(ctx.extras["compiled_memory"])
        if not isinstance(rows, list):
            raise ValueError("Arbiter must return a JSON list")
        entries = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"kind", "content", "evidence", "salience", "tags"}:
                raise ValueError("Invalid memory fields")
            if row["kind"] not in {"fact", "preference", "decision", "open_loop", "skill"}:
                raise ValueError("Invalid memory kind")
            if not isinstance(row["content"], str) or not row["content"].strip() or not isinstance(row["evidence"], str):
                raise ValueError("Memory content and evidence must be strings")
            score = row["salience"]
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Salience must be between 0 and 1")
            if not isinstance(row["tags"], list) or not all(isinstance(tag, str) for tag in row["tags"]):
                raise ValueError("Tags must be strings")
            entries.append(MemoryEntry(**row))
        merged = {(row.kind, row.content): row for row in ctx.agent.memory}
        merged.update({(row.kind, row.content): row for row in entries})
        ctx.agent.memory.replace_all(list(merged.values()))
        ctx.extras["memory_entries"] = entries

    def pipeline(self, agent):
        return [AdversarialNode(
            proposer=lambda ctx: self.role_prompt(ctx, "proposer"),
            critic=lambda ctx: self.role_prompt(ctx, "critic"),
            arbiter=lambda ctx: self.role_prompt(ctx, "arbiter"),
            commit=self.merge_memory,
        )]
