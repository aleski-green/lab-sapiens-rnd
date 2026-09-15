agentpy spec

we have custome ui ready for agent, chat, config, dashboard and etc
now we need python sdk to later integrate with ui and .ipynb to test it
idea is to have permanent agent running by cron every k minutes

basically loops: awake - short heartbit like cicles and circa generally longer cicles with two assynchronous outcomes: learning which is internal memory consolidation and morphosis scripts update of the agent itself within human/system defined contrsaints (like token budget per loop and per sprint)
sprint is a long term cicle connected with calendar days
project is long term cicle connected with calendar days, weeks, months and certain KPIs, OKRs, Goals

awake is an essentially python script which is itself part of changable adaptive (via morphosis) config
we can't spawn llm session every k minutes for no reason so must be set of rules in it to decide should llm be spawned or .py is enought 

agents can save artifacts such as documents, datasets, dashboards and etc into shared system between other agents and admin user called corpora

but main idea is that agent is defined by its python scripts and inner state, persistent state as json, there are constraints on size of agent persistable state, but all forgoten data will just go to archive which is also part of corpora

cleaning archive must not break the operations because we are assuming archived data is no longer relevant to agentic efficency and permomance

so an agent is its runtime semantic memory called memx and set of scripts morphos (body)

in the first iteration it's clear that agent has separate llms (codex cli) running in parralel and assynchroneusly
one is to only maintain conversation with user-admin
another is to reason over current agentic state, tasks, goals
another is to reason about morphosis
another is to reason about consolidation
another is to reason about corpora
another is to reAct for certain tasks and operations

one or a few llms must be designed to provide some level of friction to what user asked for or project goal is about
one or a few llms must be always live and act as 'doubting' engine following philosophy: better to not act if not neccesarry rather than produce slop and be aggreable
one llm must be aestetic controller focusing on keeping everything organized and follow minimalizm and best practices
one llm must be explorer with internet access and proactivelly follow and suggest best practices and efficiency

one llm must be focused on proactivity and initiatives
another llm must be focused on measuremnts of value created and evaluations

for consolidation and for morphosis we have to have adversarial process where we have three llms proposer, critic and arbiter
so those important for agent's integrity components would be updated reasonably

llms are ephimerial entities and all integrity of an agent is its memx semantic memory

user can stear agent by updating manifests directly updating manifest .md files or communicating in chat


corpora itself is a hierarchical organization of agents
director is main point of interaction and feedback 
other specialized agents are responsible for teams and departments
it's needed because each agent can assynchronously cover a certain scope of work, ops and project
so if needed agents can ask each other for information, clarification and even judgments



