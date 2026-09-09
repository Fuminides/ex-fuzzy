## Multi-agent delegation

Use subagents proactively when doing so meaningfully improves speed, context efficiency, or solution quality. Since Astra model uses a lot of tokens, it is preferable to use subagents of less greedy models that can do the job.

Treat the root agent as the orchestrator and integrator. The root agent should retain responsibility for:

* understanding the overall task and constraints;
* deciding architecture and cross-cutting design choices;
* decomposing the work;
* resolving disagreements between agents;
* integrating changes;
* performing final verification.

Delegate aggressively when work can be performed independently, especially:

* repository exploration and locating relevant code;
* tracing separate subsystems or execution paths;
* documentation/API research;
* test execution and failure analysis;
* log analysis;
* independent bug hypotheses;
* code review, security review, and edge-case analysis;
* independent implementation tasks with clearly separated files or interfaces.

Prefer parallel read-only exploration before implementation for non-trivial tasks.

Do not delegate merely to create more agents. Keep work in the root agent when:

* delegation would require duplicating substantial context;
* coordination/delegation cost is likely to exceed the useful work.

When delegating:

1. Give each subagent one bounded objective and a clear deliverable.
2. Prefer the cheapest model/reasoning level capable of completing that objective reliably. Hwoever, note that we have good Opus rates so we can be a little bit more conservative here whem in doubt.
3. Run independent subtasks in parallel.
4. Avoid having multiple agents modify overlapping files unless there is a compelling reason.
5. Ask subagents to return concise findings with file paths, symbols, evidence, commands run, and unresolved uncertainties rather than raw logs.
6. Wait for relevant subagents before making decisions that depend on their results.
7. Critically evaluate subagent conclusions rather than accepting them automatically.
8. When integrating work, have the root agent perform the final consistency check and run the appropriate tests.

Do not recursively spawn agents unless the child task itself contains genuinely independent substantial work.

## Cross-model delegation

Claude Opus is available through the `opus-delegate` skill.

Astra remains the orchestrator and final decision-maker. However, Opus offers good coding capabilities that can be used to maximise the work done per token.
