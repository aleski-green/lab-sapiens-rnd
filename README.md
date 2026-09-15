# AgentPy

A local Python SDK for persistent agents: semantic memory (**memx**), editable
behavior (**morphos**), and temporary LLM workers. The notebook and a future UI use
the same runtime. The design follows [the human spec](human_specs/agentpy_spec_human.md).

## Start in the notebook

Run `./start_jupyter.sh`, open `agentpy.ipynb`, and restart its kernel once after
updating the SDK. Python 3.9+ and an authenticated `codex` CLI are required.
The runtime uses only Python's standard library and supports macOS/Linux.

```python
from agentpy import AgentPy
from agentpy_codex import CodexFactory
from config import Config

agent = AgentPy.open(
    agid="notebook",
    config=Config(),
    factory=CodexFactory(event_sink=lambda message: None),
)

job = agent.tell("Remember: keep my replies concise.")
await agent.run()
print(agent.result(job))

job = agent.consolidate()
await agent.run()
agent.check(job)
print(agent.memory.render())
```

State saves automatically after each transition. Reopen with the same `agid` to
restore it. Omit `agid` to generate a new ID, available as `agent.agid`.
Opening an agent starts no LLM or background process. `tell` and `submit` queue
work; `run` drains the admitted jobs. `check` raises if a job failed or is blocked.

## What to edit

| File | Purpose |
| --- | --- |
| `config.py` | Roles, flows, schedules, memory schema, cheap wake-up rules |
| `prompts/*.md` | Complete, explicit prompt text for each role |
| `.agentpy/agents/<agid>/manifests/*.md` | Human instructions, reread for each run |
| `agentpy.ipynb` | Open an agent, chat, consolidate, inspect essentials |

`agent.set_manifest("identity", "Your instructions...")` writes a manifest.
Configuration selects a memory dataclass and can override `parse_memory` to validate
a different schema. Runtime parsing and storage code stays out of the notebook.

Prompt placeholders are explicit: `{context}`, `{task}`, `{memory}`, `{manifests}`,
`{chat}`, `{goals}`, `{tasks}`, `{notes}`, `{proposal}`, `{critique}`, `{last}`,
`{body}`, `{corpora}`, and `{directory}`. Only requested placeholders are inserted;
the runtime adds no role instructions. Escape literal JSON braces as `{{` and `}}`.

Independent jobs run concurrently. Steps inside a flow run in order:

```python
from agentpy import Flow, Python, Role

# Add to Config.roles and Config.flows:
roles = {
    "draft": Role("Propose a solution: {task}"),
    "review": Role("Review this proposal: {last}"),
}
flows = {
    "experiment": Flow(("draft", Python(lambda ctx: ctx["last"].strip()), "review")),
}
```

The configured roles cover chat, reasoning, doubt, exploration, aesthetics,
initiative, evaluation, corpora, execution, learning, and morphosis. They are
standing capabilities; an LLM is created only when its flow is requested.

## Scheduling and recovery

`await agent.tick()` checks mail, runs `Config.awake`, and drains queued work.
The default heartbeat is every ten minutes. A daily circa queues learning and
morphosis only when new experience exists. Due tasks are dispatched once.
An idle heartbeat makes no LLM call. Additional review/initiative rules belong in
`Config.awake`; the default does not launch every role each day.

```bash
python3 -m agentpy --agent notebook --status
python3 -m agentpy --agent notebook           # one tick, suitable for cron
python3 -m agentpy --agent notebook --serve   # persistent foreground runner
```

For a macOS service that survives terminal closure and restarts after a crash:

```bash
python3 -m agentpy --agent notebook --service-file "$HOME/Library/LaunchAgents/local.agentpy.notebook.plist"
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/local.agentpy.notebook.plist"
```

Generating a service file does not start it. Run these commands from this repo
using the Python environment where Codex is available. No service is installed by
opening the notebook. Service logs live in `.agentpy/service-logs/`.

Jobs and budget reservations are saved before workers start. A per-agent process
lock prevents overlapping runners; separate processes can safely submit messages.
On recovery, unfinished calls become `interrupted`. They are not automatically
replayed because their external effects may already have happened. Inspect them,
then use `agent.retry(job)` or `agent.cancel(job)`. Budget-blocked jobs are considered
again on the next run. Notebook cancellation waits for current bounded worker calls
to finish before releasing the runner lock.

## Memory, limits, and budgets

Workers receive snapshots. Only the runtime commits state changes, using atomic
JSON replacement under a lock. Conflicting memory/body revisions reject stale
results as `conflict`, available for explicit retry. A concurrent learning and
morphosis run can therefore require rerunning one proposal on fresher state.

Learning is proposer → critic → arbiter. Its result has `upsert` and `forget` lists.
Existing memory IDs support corrections/removals; unrelated entries survive.
Invalid changes never partially update memory. Prompts define what is worth keeping.

Limits are stored inside agent state: 1,000 records per active history, at most
1,000 memory entries, 1 MB total serialized state, and 60,000 characters per prompt.
Old chat/events/completed job summaries go to archive. Required tasks and memory
are not silently evicted; oversized active state is rejected. Large outputs belong
in active corpora artifacts. Archive deletion does not prevent agent recovery.

The host reserves 16,000 tokens per LLM call, admits up to 128,000 per run and
1,000,000 per calendar sprint, and reconciles reported input/output usage. Missing
usage is charged at the full reservation. The sprint calendar persists across
restarts; adaptive config cannot reset it. `Limits` is a host-side override.

These are admission/accounting limits, not a hard token cutoff inside Codex. A
single provider call can exceed its reservation; actual usage is still charged,
and later calls are stopped when the flow allowance is exhausted. Calls have a
120-second deadline by default. Set `CodexFactory(timeout_seconds=...)` to change it.

## Morphosis

`agent.submit("morphosis", "Improve a specific behavior")` queues its debate.
The arbiter returns full replacement files in a `files` object, or an empty object
for no change. Candidate versions get Python syntax checks, isolated-process
configuration/flow validation, and any host-specified `MorphPolicy.checks` commands.
Successful versions activate at the next run boundary. `agent.rollback()` restores
the previous version. Adaptive imports should be relative, such as
`from .behaviors import awake`.

Default editable files are `config.py`, `behaviors.py`, and `prompts/*.md`.
`MorphPolicy` is supplied by the host, separately from adaptive configuration.
Before the first activation, behavior comes from the supplied Config and source
files. Afterwards, the validated version under `morphos/versions/` is authoritative;
editing the original config does not override that active version.

Codex executes locally with unrestricted filesystem access, as requested. Prompt
instructions and morphosis path checks govern this runtime's commit path; they do
not sandbox arbitrary local tools or Python. Default role prompts prohibit tool
use except exploration and explicit execution. Behavioral test commands should be
added to host policy for substantive self-modifications.

## Corpora, projects, and UI integration

```python
agent.corpora.put("research/result.md", "Findings...")
agent.corpora.register(agent.agid, parent="director", scope="Research")
message = agent.send("director", "Findings are ready")
# Reply with agent.send(sender, text, reply_to=message_id).

project = agent.add_project("Memory study", starts="2026-09-15", ends="2026-10-15",
                            goals=["Improve preference recall"], metrics={"recall": 0.0})
agent.add_task("Evaluate recall", due="2026-09-16T09:00:00+04:00", flow="evaluate", project=project)
agent.measure(project, "recall", 0.85)
```

Register a parent before assigning children. Durable mailboxes ingest messages
before acknowledging delivery. Incoming requests run the chat flow and send its
answer back through a durable outbox. Replies become experience notes, avoiding
automatic reply loops. Delivery receipts stay in active corpora coordination
storage, separate from the deletable archive. Agents share the same corpora root by default;
pass `Corpora(path)` explicitly when agents use different state roots. Directory
hierarchy is organizational metadata, not an access-control system.

The UI can call `tell`, `submit`, `status`, `result`, and `events(after=sequence)`.
State reads return snapshots. `sessions` exposes compact run summaries;
`transcript(job)` loads detailed logs only on demand and returns an empty list if
the archive was removed. Event polling is bounded: fetch fresh `state` if your
cursor is older than the retained event window. No HTTP/UI server is introduced.

## Development

```bash
python3 -m pip install -e .
python3 -m unittest test_runtime test_adversarial test_codex_process -v
```

Tests use temporary stores and scripted workers. The original synchronous
`AgentPy(config, factory)` and `AdversarialConfig` remain available for older
experiments. Use `AgentPy.open` for the persistent lifecycle. Existing kernel-only
state is not automatically migrated; the old `agentpy_state.py` helper remains
legacy code.
