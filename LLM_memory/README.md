# LLM development memory

Read [DEVELOPMENT.md](DEVELOPMENT.md) first. These notes preserve repository
conventions and evidence useful for future development. Check dated claims
against the current code. Historical plans and session approvals do not authorize
new work, and deferred ideas are not promised features.

- [DEVELOPMENT.md](DEVELOPMENT.md): code map, conventions, and essential invariants.
- [SPEED_UP.md](SPEED_UP.md): dated implementation history, exactness pitfalls,
  benchmark results, and reproduction commands. Commands run from the repository root.
- [SPEED_UP_REVIEW.md](SPEED_UP_REVIEW.md): decisions and evidence for the 36
  performance proposals, including rejected approaches.
- [DELEGATION.md](DELEGATION.md): shared guidance for bounded parallel work.

Keep durable decisions, evidence, and unresolved constraints here. Remove stale
instructions and duplicated tutorials rather than accumulating session transcripts.
User-facing documentation belongs in `docs/source/`; the performance guide is
[Training Performance](../docs/source/user-guide/training-performance.rst).
