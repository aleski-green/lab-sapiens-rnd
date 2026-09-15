# Agentpy

AgentPy is a notebook-based agent runtime with configurable LLM flows. It owns manifests, memory, chat, and session logs; config classes supply prompts, schemas, and Python pipeline functions. Codex workers can start fresh sessions or resume existing ones.

## Contents

- `agentpy.ipynb` — interactive examples, consolidation test, and adversarial-node demo.
- `agentpy/` — interfaces, runtime, and reusable `LLLamb` / `AdversarialNode` steps.
- `config.py` — base prompts, memory schema, and pipeline.
- `adversarial_config.py` — proposer, critic, arbiter prompts and validated memory merge.
- `agentpy_codex.py` — streaming local Codex execution and session resume.
- `agentpy_state.py` — legacy JSON persistence helper; not wired into the current runtime.

## Requirements

- Python 3.9+
- An authenticated `codex` CLI available on `PATH`
- JupyterLab installed in `../.jupyter-venv`

Start the local notebook server with:

```bash
./start_jupyter.sh
```

It opens a Terminal window that keeps JupyterLab running at
`http://127.0.0.1:8888/lab`. Keep that window open while using notebooks,
then open `agentpy.ipynb`.

## Usage

```python
from agentpy import AgentPy
from agentpy_codex import CodexFactory
from adversarial_config import AdversarialConfig

agent = AgentPy(config=AdversarialConfig(), factory=CodexFactory())
result = agent.handle("Design a minimal memory experiment.")
print(agent.memory.render())
print(result.extras["debate"]["critique"])
```

The adversarial node snapshots context, calls proposer → critic → arbiter with separate sessions, then parses and validates the arbiter's JSON before merging memory. Prompt lambdas are explicitly configured; roles do not silently add instructions. Full proposal and critique text is passed to later roles. Each run gets unique session keys.

Merge preserves unrelated entries and deduplicates exact `(kind, content)` matches. It does not automatically resolve semantic contradictions. Invalid output raises an error without changing memory. The demo deliberately omits automatic user replies and the base pipeline's additional consolidation call.

## Persistence

Current AgentPy memory and logs are in-process only; kernel restart loses them. Saved notebook outputs are not restored agent state. The earlier JSON helper and 1,000-invocation retention policy are not integrated into this architecture.

## Tests

Run `python -m unittest test_adversarial test_codex_process -v` for deterministic pipeline, failure, timeout, and interrupt tests. The final notebook section runs the real local backend and checks memory retention and prompt piping.

Codex calls have a 120-second per-call deadline, adjustable with `CodexFactory(timeout_seconds=...)`. A timeout or notebook interrupt stops the spawned process group. Failed calls do not reach the memory merge step. Live JSON events follow the [Codex non-interactive interface](https://learn.chatgpt.com/docs/non-interactive-mode).

## Safety

`agentpy_codex.py` runs Codex with `--dangerously-bypass-approvals-and-sandbox`. An invocation can therefore make unrestricted changes on the local machine. Use it only in a workspace and environment you trust.
