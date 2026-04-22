# Docker

Container scaffolding for eval_report_gen.

Design goals:

- the agent container sees only staged public studies and writable outputs
- the eval container sees public studies, private references, and read-only agent outputs
- agent and eval use separate images and separate mount sets

Important mounting rule:

- agent container: mount `public/`, never `private/`
- eval container: mount both `public/` and `private/`

Key files:

- `orchestrator.py`
- `agent/Dockerfile.agent`
- `eval/Dockerfile.eval`
- `tests/test_orchestrator.py`
