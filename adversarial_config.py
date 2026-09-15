"""Compatibility config for the original synchronous flow experiments."""
from agentpy import AdversarialNode
from agentpy.settings import FlowConfig, Prompt, merge_memory


class AdversarialConfig(FlowConfig):
    proposer = Prompt("Propose a concise solution.\n{context}\nTask: {task}")
    critic = Prompt("Critique the proposal.\n{context}\nTask: {task}\nProposal: {proposal}")
    arbiter = Prompt("""Extract durable memory additions. Return JSON only: a list of
objects with kind, content, evidence, salience, tags. Kinds: fact, preference,
decision, open_loop, skill. Salience: 0 to 1. Tags: strings. Do not use tools.
{context}
Task: {task}
Proposal: {proposal}
Critique: {critique}""")
    memory_prompt = Prompt("""Extract durable memory additions. Return JSON only: a list
of objects with kind, content, evidence, salience, tags. Kinds: fact, preference,
decision, open_loop, skill. Salience: 0 to 1. Tags: strings. Do not use tools.
{context}""")
    default_prompt = Prompt("{context}\nTask: {task}")
    flow = [AdversarialNode(proposer=proposer, critic=critic, arbiter=arbiter, commit=merge_memory)]
