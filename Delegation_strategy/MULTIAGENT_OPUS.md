## Delegation policy

Use subagents proactively for substantial work when they provide useful
parallelism, context isolation, or independent verification.

The main agent owns:
- overall understanding of the task;
- architecture and cross-cutting design decisions;
- decomposition of the work;
- integration of changes;
- resolution of conflicting findings;
- final correctness verification.

Prefer delegation for:
- independent repository exploration;
- tracing different subsystems;
- documentation and API research;
- test execution and failure analysis;
- debugging competing hypotheses;
- code review;
- security/performance review;
- independent implementation work with non-overlapping ownership.

Parallelize independent work whenever this materially reduces latency.

Do not delegate trivial operations such as a simple grep, reading one file,
a tiny edit, or a strongly sequential task.

Use subagents especially to keep large search results, logs, test output,
and exploratory work out of the main context.

When delegating:
- give each agent a bounded objective;
- tell it what deliverable to return;
- avoid overlapping edits;
- ask for concise findings with paths, symbols, evidence, and uncertainties;
- critically evaluate its result rather than blindly accepting it.

For substantial coding tasks prefer:

parallel exploration
→ main agent chooses architecture
→ parallel independent implementation where appropriate
→ main agent integrates
→ independent review/testing
→ main agent performs final fixes and verification.
