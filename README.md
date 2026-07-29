# egraph

ManiSkill planning experiments (currently PushT / sim planner).

## Quick start

```bash
uv sync
uv run python main.py
```

## Modules

| Module | Role |
|--------|------|
| `main.py` | Plan and render a PushT-v1 episode with `SimPlanner` |
| `planner.py` | `SimPlanner` search / plan execution |
| `scripts/execute_plan.py` | Helper to execute a saved plan |
| `scripts/pushT_test.py` | PushT smoke script |

## Archived

Table-bussing / ACRONYM dishware code lives under [`archive/table_bussing/`](archive/table_bussing/). See that folder's `RESTORE.md` to bring it back.
