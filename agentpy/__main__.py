"""Local runner: python -m agentpy --agent notebook [--serve]."""
import argparse
import asyncio
import importlib.util
from pathlib import Path
import sys

from .agent import AgentPy


def load_config(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("_agentpy_user_config", path,
                                                 submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Config()


async def serve(agent):
    while True:
        await agent.tick()
        # Poll mail every ten seconds; awake() enforces the slower configured heartbeat.
        await asyncio.sleep(10)


def main():
    parser = argparse.ArgumentParser(description="Run a persistent local AgentPy agent")
    parser.add_argument("--agent", required=True, help="Stable agent ID")
    parser.add_argument("--root", default=".agentpy")
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--serve", action="store_true", help="Keep checking for work")
    parser.add_argument("--status", action="store_true", help="Show essentials without running LLMs")
    parser.add_argument("--service-file", help="Write a launchd plist without starting the agent")
    args = parser.parse_args()
    if args.service_file:
        from .service import write_service
        write_service(args.service_file, agid=args.agent, root=args.root, config=args.config)
        print(f"Service configuration written to {args.service_file}")
        return
    from agentpy_codex import CodexFactory
    agent = AgentPy.open(config=load_config(args.config), factory=CodexFactory(event_sink=lambda _: None),
                         agid=args.agent, root=args.root, source=Path(args.config).resolve().parent)
    if not args.status:
        try:
            asyncio.run(serve(agent) if args.serve else agent.tick())
        except KeyboardInterrupt:
            pass
    print(agent.status())


if __name__ == "__main__":
    main()
