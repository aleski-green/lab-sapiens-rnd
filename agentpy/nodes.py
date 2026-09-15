"""Composable LLM nodes. Prompts and commit policy belong to the config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

from .interfaces import LLMSpec, PipeContext, PromptFn, Spawn, Step


@dataclass
class LLLamb(Step):
    key: str
    prompt: PromptFn
    output: str
    spec: LLMSpec | None = None

    def run(self, ctx: PipeContext) -> PipeContext:
        Spawn(self.key, spec=self.spec, prompt=self.prompt).run(ctx)
        ctx.extras[self.output] = ctx.last_output
        return ctx


@dataclass
class AdversarialNode(Step):
    proposer: PromptFn
    critic: PromptFn
    arbiter: PromptFn
    commit: Callable[[PipeContext], None]
    key: str = "debate"

    def run(self, ctx: PipeContext) -> PipeContext:
        # Freeze shared context before any role starts; preserve logs on reruns.
        run_key = f"{self.key}/{uuid4().hex[:12]}"
        work = PipeContext(agent=ctx.agent, payload=ctx.payload, extras={
            "context": ctx.agent.inject(), "run_key": run_key,
        })
        for role, prompt, output in (
            ("proposer", self.proposer, "proposal"),
            ("critic", self.critic, "critique"),
            ("arbiter", self.arbiter, "compiled_memory"),
        ):
            LLLamb(f"{run_key}/{role}", prompt, output, LLMSpec(role=role)).run(work)
        self.commit(work)
        ctx.extras[self.key] = work.extras
        ctx.last_output = work.last_output
        ctx.last_llm = work.last_llm
        ctx.session_key = work.session_key
        return ctx
