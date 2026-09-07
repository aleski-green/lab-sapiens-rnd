# Agentpy

Agentpy is a small notebook-based agent wrapper around the local `codex exec` CLI. Each task starts a fresh Codex worker while `Agentpy` keeps the durable context and invocation history on disk.

## Contents

- `agentpy.ipynb` — the `Agentpy` class and an interactive example.
- `agentpy_codex.py` — local Codex execution.
- `agentpy_state.py` — JSON persistence for agent state.

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
agent = Agentpy(
    context="You are a helpful coding agent.",
    manifest="Work only inside the current project.",
    contextUpdPrompt="Update context with durable facts from the last task.",
)

agent.llmrun("Inspect this project and summarize its structure.")
print(agent.invocations[-1])

agent.llmupd()
print(agent.context)

# Equivalent to llmrun() followed by llmupd().
agent.llmrunupd("Record the project's main components.")
```

`llmrun()` records the latest invocation and answer, retaining an internal sliding window of 1,000 records. `llmupd()` runs `contextUpdPrompt` exactly as supplied, stores its answer in `context`, and clears the temporary `last_invocation` and `last_result` fields. All three agent methods mutate state and return nothing.

## Persistence

Every run and update saves JSON state at:

```text
<workdir>/.agentpy/<agid>.json
```

Restore an existing agent by its ID:

```python
agent = Agentpy("", "", "", agid="ag_your_saved_id")
agent.load()
```

The state file includes the agent ID, context, manifest, update prompt, invocation history, and any pending invocation/result pair. `.agentpy/` is ignored by Git.

## Safety

`agentpy_codex.py` runs Codex with `--dangerously-bypass-approvals-and-sandbox`. An invocation can therefore make unrestricted changes on the local machine. Use it only in a workspace and environment you trust.
