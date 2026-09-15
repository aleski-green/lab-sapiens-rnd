"""Generate a macOS launchd service; generation does not install or start it."""
import os
from pathlib import Path
import plistlib
import sys

from .storage import atomic_bytes


def write_service(path, *, agid, root, config):
    workspace = Path(config).resolve().parent
    logs = Path(root).resolve() / "service-logs"
    logs.mkdir(parents=True, exist_ok=True)
    if not agid or not all(c.isalnum() or c in "_-" for c in agid):
        raise ValueError("Service agent ID must contain only letters, digits, _ or -")
    payload = {
        "Label": f"local.agentpy.{agid}",
        "ProgramArguments": [sys.executable, "-m", "agentpy", "--agent", agid,
                             "--root", str(Path(root).resolve()), "--config", str(Path(config).resolve()),
                             "--serve"],
        "WorkingDirectory": str(workspace),
        "EnvironmentVariables": {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                                 "PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 30,
        "StandardOutPath": str(logs / f"{agid}.out.log"),
        "StandardErrorPath": str(logs / f"{agid}.err.log"),
    }
    atomic_bytes(Path(path).expanduser().resolve(), plistlib.dumps(payload))
