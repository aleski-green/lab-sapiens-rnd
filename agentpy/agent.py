"""Generic AgentPy runtime + in-memory stores.

Personality is not here — inject a Config and an LLMFactory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from json import dumps
from typing import Any, Iterator, Sequence
from uuid import uuid4

from agentpy.interfaces import (
    Agent,
    AgentConfig,
    Chat,
    LLM,
    LLMFactory,
    LLMSpec,
    Manifests,
    Memory,
    Message,
    PipeContext,
    SessionLog,
    Sessions,
    T,
    Turn,
)


# ---------------------------------------------------------------------------
# In-memory stores (swap later for JSON / sqlite without touching AgentPy)
# ---------------------------------------------------------------------------


@dataclass
class DictManifests(Manifests):
    data: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, key: str) -> str:
        return self.data[key]

    def __setitem__(self, key: str, value: str) -> None:
        self.data[key] = value

    def keys(self) -> Sequence[str]:
        return list(self.data)

    def render(self) -> str:
        if not self.data:
            return "(none)"
        return "\n\n".join(f"## {key}\n{value}" for key, value in self.data.items())


@dataclass
class ListMemory(Memory[T]):
    schema: type
    items: list[T] = field(default_factory=list)

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def add(self, item: T) -> None:
        if not isinstance(item, self.schema):
            raise TypeError(f"memory row must be {self.schema.__name__}, got {type(item).__name__}")
        self.items.append(item)

    def extend(self, items: Sequence[T]) -> None:
        for item in items:
            self.add(item)

    def replace_all(self, items: Sequence[T]) -> None:
        self.items = []
        self.extend(items)

    def clear(self) -> None:
        self.items = []

    def render(self) -> str:
        if not self.items:
            return "(empty)"
        blocks = []
        for item in self.items:
            render = getattr(item, "render", None)
            if callable(render):
                blocks.append(render())
            elif is_dataclass(item) and not isinstance(item, type):
                blocks.append(dumps(asdict(item), ensure_ascii=False))
            else:
                blocks.append(str(item))
        return "\n".join(blocks)


@dataclass
class ListChat(Chat):
    messages: list[Message] = field(default_factory=list)
    window: int = 50

    def append(self, role: str, content: str) -> Message:
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        if len(self.messages) > self.window:
            self.messages = self.messages[-self.window :]
        return msg

    def recent(self, n: int | None = None) -> Sequence[Message]:
        if n is None:
            return list(self.messages)
        return self.messages[-n:]

    def clear(self) -> None:
        self.messages = []

    def render(self) -> str:
        if not self.messages:
            return "(empty)"
        return "\n".join(f"[{m.role}] {m.content}" for m in self.messages)


@dataclass
class DictSessions(Sessions):
    data: dict[str, SessionLog] = field(default_factory=dict)

    def __getitem__(self, key: str) -> SessionLog:
        return self.data[key]

    def get(self, key: str) -> SessionLog | None:
        return self.data.get(key)

    def keys(self) -> Sequence[str]:
        return list(self.data)

    def open(self, key: str, llm: LLM) -> SessionLog:
        log = SessionLog(key=key, llm_id=llm.id, spec=llm.spec)
        self.data[key] = log
        return log

    def append_turn(self, key: str, turn: Turn) -> None:
        self.data[key].turns.append(turn)

    def transcript(self, key: str) -> str:
        return self.data[key].render()

    def render(self) -> str:
        if not self.data:
            return "(none)"
        lines = []
        for key, log in self.data.items():
            last = log.turns[-1].content[:120] if log.turns else ""
            lines.append(f"- {key}  llm={log.llm_id}  turns={len(log.turns)}  last={last!r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Echo backend — so the OOP graph is instantiable without a real LLM
# ---------------------------------------------------------------------------


@dataclass
class EchoLLM(LLM):
    spec: LLMSpec
    id: str = field(default_factory=lambda: f"llm_{uuid4().hex[:8]}")

    def complete(self, prompt: str) -> str:
        return f"[echo:{self.spec.role}] {prompt[-400:]}"


class EchoFactory(LLMFactory):
    def spawn(self, spec: LLMSpec) -> LLM:
        return EchoLLM(spec=spec)

    def iterate(self, session: SessionLog) -> LLM:
        return EchoLLM(spec=session.spec, id=session.llm_id)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class AgentPy(Agent):
    @classmethod
    def open(cls, *, config, factory, agid=None, **options):
        """Create or restore a persistent agent; background work starts explicitly."""
        from .runtime import PersistentAgent
        return PersistentAgent(config=config, factory=factory, agid=agid, **options)

    def __init__(
        self,
        config: AgentConfig,
        factory: LLMFactory | None = None,
        *,
        manifests: dict[str, str] | Manifests | None = None,
        agid: str | None = None,
    ) -> None:
        self.agid = agid or f"ag_{uuid4().hex[:12]}"
        self.config = config
        self.factory = factory or EchoFactory()
        self.manifests = (
            manifests
            if isinstance(manifests, Manifests)
            else DictManifests(data=dict(manifests or {}))
        )
        self.memory: Memory[Any] = ListMemory(schema=config.memory_schema)
        self.chat = ListChat()
        self.sessions = DictSessions()

    def spawn(self, key: str, spec: LLMSpec | None = None) -> LLM:
        llm = self.factory.spawn(spec or self.config.default_spec())
        self.sessions.open(key, llm)
        return llm

    def iterate(self, key: str) -> LLM:
        log = self.sessions[key]
        return self.factory.iterate(log)

    def inject(self, *, session_key: str | None = None) -> str:
        parts = [
            ("# Manifests", self.manifests.render()),
            ("# Memory", self.memory.render()),
            ("# Chat", self.chat.render()),
        ]
        if session_key is not None:
            parts.append((f"# Session {session_key}", self.sessions.transcript(session_key)))
        else:
            parts.append(("# Sessions", self.sessions.render()))
        return "\n\n".join(f"{title}\n{body}" for title, body in parts)

    def handle(self, user_message: str) -> PipeContext:
        self.chat.append("user", user_message)
        return self.run_pipeline(PipeContext(agent=self, payload=user_message))
