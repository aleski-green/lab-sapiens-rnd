"""AgentPy contracts — declare these before any implementation.

Object graph
------------
    AgentPy                     orchestrator (generic runtime)
      .manifests                user-authored identity / rules
      .memory                   distilled experience (schema from Config)
      .chat                     recent user <-> agent dialogue
      .sessions                 {key: SessionLog} of LLM transcripts
      .factory                  LLMFactory.spawn | .iterate
      .config                   personality: schemas, prompts, lambdas, IO

    Config (config.py)          the only file that should change per-agent
      memory_schema             dataclass for one memory row
      consolidate               rewrite memory from chat + sessions
      ask_user / respond_to_user
      pipeline                  [lambda, Spawn, lambda, Iterate, ...]

A pipeline is a sequence of Steps and plain callables. Spawn/Iterate are
LLM steps; everything else is a lambda that reshapes PipeContext.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Generic, Iterator, Sequence, TypeVar


# ---------------------------------------------------------------------------
# Data (not ABC — these are the wire shapes)
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class LLMSpec:
    """How to construct an LLM worker. Factory-specific keys go in extra."""

    model: str = "default"
    role: str = "worker"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Turn:
    role: str  # user | assistant | system | tool | lambda
    content: str
    ts: str = field(default_factory=_utcnow)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    role: str  # user | agent | system
    content: str
    ts: str = field(default_factory=_utcnow)


@dataclass
class SessionLog:
    key: str
    llm_id: str
    spec: LLMSpec
    turns: list[Turn] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"## session:{self.key}  llm={self.llm_id}  role={self.spec.role}"]
        for turn in self.turns:
            lines.append(f"[{turn.role}] {turn.content}")
        return "\n".join(lines)


@dataclass
class PipeContext:
    """Mutable bag that flows through the pipeline."""

    agent: Agent
    payload: Any = None
    session_key: str | None = None
    last_llm: LLM | None = None
    last_output: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


PipeFn = Callable[[PipeContext], PipeContext]
PromptFn = Callable[[PipeContext], str]
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Context injectables — anything Agent.inject() can dump into an LLM prompt
# ---------------------------------------------------------------------------


class Renderable(ABC):
    @abstractmethod
    def render(self) -> str:
        """Markdown (or plain text) block for LLM context."""


class Manifests(Renderable, ABC):
    """Named documents supplied by the user (identity, rules, tools, ...)."""

    @abstractmethod
    def __getitem__(self, key: str) -> str: ...

    @abstractmethod
    def __setitem__(self, key: str, value: str) -> None: ...

    @abstractmethod
    def keys(self) -> Sequence[str]: ...


class Memory(Renderable, ABC, Generic[T]):
    """Experience store. Row type is Config.memory_schema."""

    @abstractmethod
    def __iter__(self) -> Iterator[T]: ...

    @abstractmethod
    def add(self, item: T) -> None: ...

    @abstractmethod
    def extend(self, items: Sequence[T]) -> None: ...

    @abstractmethod
    def replace_all(self, items: Sequence[T]) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...


class Chat(Renderable, ABC):
    """Sliding window of the conversation with the human."""

    @abstractmethod
    def append(self, role: str, content: str) -> Message: ...

    @abstractmethod
    def recent(self, n: int | None = None) -> Sequence[Message]: ...

    @abstractmethod
    def clear(self) -> None: ...


class Sessions(Renderable, ABC):
    """Map of session_key -> LLM transcript."""

    @abstractmethod
    def __getitem__(self, key: str) -> SessionLog: ...

    @abstractmethod
    def get(self, key: str) -> SessionLog | None: ...

    @abstractmethod
    def keys(self) -> Sequence[str]: ...

    @abstractmethod
    def open(self, key: str, llm: LLM) -> SessionLog: ...

    @abstractmethod
    def append_turn(self, key: str, turn: Turn) -> None: ...

    @abstractmethod
    def transcript(self, key: str) -> str:
        """Full log for one session key."""


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


class LLM(ABC):
    """One worker bound to one session. The backend may keep extra state
    behind .id; AgentPy only stores the transcript in sessions[key]."""

    id: str
    spec: LLMSpec

    @abstractmethod
    def complete(self, prompt: str) -> str: ...


class LLMFactory(ABC):
    @abstractmethod
    def spawn(self, spec: LLMSpec) -> LLM:
        """Brand-new worker / backend session."""

    @abstractmethod
    def iterate(self, session: SessionLog) -> LLM:
        """Rehydrate the worker that owns this session log."""


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


class Step(ABC):
    @abstractmethod
    def run(self, ctx: PipeContext) -> PipeContext: ...


def _resolve_prompt(ctx: PipeContext, prompt: PromptFn | str | None) -> str:
    if prompt is None:
        return ctx.agent.config.build_prompt(ctx)
    if isinstance(prompt, str):
        return prompt
    return prompt(ctx)


@dataclass
class Spawn(Step):
    """factory.spawn → complete → write turns into sessions[key]."""

    key: str
    spec: LLMSpec | None = None
    prompt: PromptFn | str | None = None

    def run(self, ctx: PipeContext) -> PipeContext:
        llm = ctx.agent.spawn(self.key, self.spec)
        return _call_llm(ctx, key=self.key, llm=llm, prompt=self.prompt)


@dataclass
class Iterate(Step):
    """factory.iterate on an existing sessions[key] → complete again."""

    key: str
    prompt: PromptFn | str | None = None

    def run(self, ctx: PipeContext) -> PipeContext:
        llm = ctx.agent.iterate(self.key)
        return _call_llm(ctx, key=self.key, llm=llm, prompt=self.prompt)


def _call_llm(
    ctx: PipeContext,
    *,
    key: str,
    llm: LLM,
    prompt: PromptFn | str | None,
) -> PipeContext:
    text = _resolve_prompt(ctx, prompt)
    ctx.agent.sessions.append_turn(key, Turn(role="user", content=text))
    output = llm.complete(text)
    # A provider may only learn its durable session ID after the first call.
    # Persist it before a later Iterate step asks the factory to rehydrate it.
    ctx.agent.sessions[key].llm_id = llm.id
    ctx.agent.sessions.append_turn(key, Turn(role="assistant", content=output))
    ctx.session_key = key
    ctx.last_llm = llm
    ctx.last_output = output
    return ctx


# ---------------------------------------------------------------------------
# Personality — implemented in config.py
# ---------------------------------------------------------------------------


class AgentConfig(ABC):
    """Swap this class to change schema, prompts, pipes, and user IO.

    AgentPy stays generic. All 'what should this agent be?' lives here.
    """

    memory_schema: type

    @abstractmethod
    def default_spec(self) -> LLMSpec: ...

    @abstractmethod
    def build_prompt(self, ctx: PipeContext) -> str:
        """Default prompt for Spawn/Iterate when they don't pass one."""

    @abstractmethod
    def consolidation_prompt(self, agent: Agent) -> str: ...

    @abstractmethod
    def parse_memory(self, raw: str) -> list[Any]:
        """Turn consolidator LLM output into memory_schema instances."""

    @abstractmethod
    def consolidate(self, agent: Agent) -> None: ...

    @abstractmethod
    def ask_user(self, agent: Agent, question: str) -> str: ...

    @abstractmethod
    def respond_to_user(self, agent: Agent, message: str) -> None: ...

    @abstractmethod
    def pipeline(self, agent: Agent) -> Sequence[Step | PipeFn]: ...


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent(ABC):
    """The orchestrator interface. AgentPy is the concrete runtime."""

    agid: str
    manifests: Manifests
    memory: Memory[Any]
    chat: Chat
    sessions: Sessions
    config: AgentConfig
    factory: LLMFactory

    @abstractmethod
    def spawn(self, key: str, spec: LLMSpec | None = None) -> LLM: ...

    @abstractmethod
    def iterate(self, key: str) -> LLM: ...

    @abstractmethod
    def inject(self, *, session_key: str | None = None) -> str:
        """Assemble manifests + memory + chat + sessions for an LLM call."""

    @abstractmethod
    def handle(self, user_message: str) -> PipeContext:
        """Record the user turn and run config.pipeline()."""

    def ask_user(self, question: str) -> str:
        answer = self.config.ask_user(self, question)
        self.chat.append("agent", question)
        self.chat.append("user", answer)
        return answer

    def respond_to_user(self, message: str) -> None:
        self.chat.append("agent", message)
        self.config.respond_to_user(self, message)

    def consolidate(self) -> None:
        self.config.consolidate(self)

    def run_pipeline(self, ctx: PipeContext | None = None) -> PipeContext:
        if ctx is None:
            ctx = PipeContext(agent=self)
        for step in self.config.pipeline(self):
            ctx = step.run(ctx) if isinstance(step, Step) else step(ctx)
        return ctx
