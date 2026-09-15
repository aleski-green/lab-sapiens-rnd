import json
import unittest

from agentpy import AgentPy, EchoFactory
from adversarial_config import AdversarialConfig
from config import MemoryEntry


class ScriptedFactory(EchoFactory):
    def __init__(self, final):
        self.final = final
        self.prompts = []

    def spawn(self, spec):
        llm = super().spawn(spec)
        def complete(prompt):
            self.prompts.append(prompt)
            return {"proposer": "Full proposal", "critic": "Full critique", "arbiter": self.final, "memory": self.final}[spec.role]
        llm.complete = complete
        return llm


class AdversarialTests(unittest.TestCase):
    def make_agent(self, final):
        agent = AgentPy(AdversarialConfig(), ScriptedFactory(final))
        agent.memory.add(MemoryEntry("fact", "Existing unrelated knowledge"))
        return agent

    def test_flow_and_repeated_runs(self):
        row = dict(kind="preference", content="HPR-917", evidence="User", salience=1, tags=[])
        agent = self.make_agent(json.dumps([row]))
        result = agent.handle("Remember HPR-917")
        self.assertIn("Full proposal", agent.factory.prompts[1])
        self.assertIn("Full critique", agent.factory.prompts[2])
        self.assertIn("Full proposal", agent.factory.prompts[2])
        self.assertEqual(len(list(agent.memory)), 2)
        self.assertEqual(len(agent.sessions.keys()), 3)
        for role, sent in zip(("proposer", "critic", "arbiter"), agent.factory.prompts):
            template = getattr(agent.config, role).text
            self.assertTrue(sent.startswith(template.strip().splitlines()[0]))
        self.assertEqual(result.extras["debate"]["proposal"], "Full proposal")
        agent.handle("Remember HPR-917")
        self.assertEqual(len(list(agent.memory)), 2)
        self.assertEqual(len(agent.sessions.keys()), 6)

    def test_invalid_output_preserves_memory(self):
        for raw in ("not JSON", "{}", '[{"kind":"invented"}]'):
            agent = self.make_agent(raw)
            before = list(agent.memory)
            with self.assertRaises(ValueError):
                agent.handle("test")
            self.assertEqual(list(agent.memory), before)
            self.assertEqual(len(agent.sessions.keys()), 3)

    def test_explicit_prompt_only(self):
        from agentpy.settings import Prompt
        from agentpy import PipeContext
        agent = self.make_agent("[]")
        prompt = Prompt("Only this task: {task}")
        self.assertEqual(prompt(PipeContext(agent=agent, payload="hello")), "Only this task: hello")

    def test_standalone_consolidation_merges(self):
        row = dict(kind="preference", content="Concise replies", evidence="User", salience=1, tags=[])
        agent = self.make_agent(json.dumps([row]))
        agent.chat.append("user", "I prefer concise replies")
        agent.consolidate()
        self.assertEqual(len(list(agent.memory)), 2)
        self.assertIn("I prefer concise replies", agent.factory.prompts[0])

    def test_quiet_progress(self):
        from agentpy.settings import progress
        from contextlib import redirect_stdout
        from io import StringIO
        output = StringIO()
        with redirect_stdout(output):
            for message in ("🧠 Codex is working", "🤖 Long answer", "🧵 Session ID", "✅ Tokens..."):
                progress(message)
        self.assertEqual(output.getvalue(), "Working…\nDone.\n")


if __name__ == "__main__":
    unittest.main()
