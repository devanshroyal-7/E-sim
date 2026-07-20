# egraph

ManiSkill table-bussing environment using a curated ACRONYM dishware catalog.

## Modules

| Module | Role |
|--------|------|
| `acronym_dishware.py` | ACRONYM catalog, curated selection, mesh/physics helpers |
| `dishware_actors.py` | Shared Sapien actor construction for dishware meshes |
| `table_bussing.py` | `TableBussing-v1` env, `DishwareCounts`, and `make_env` factory |
| `render_dishware_gallery.py` | Visual QA gallery for the curated selection |
| `envs/` | Package boundary re-exporting the ManiSkill env modules |
| `environment.py` | Thin re-export of `make_env` + interactive smoke demo |
| `main.py` | Minimal entry that calls `environment.make_env` |

## Quick start

```bash
uv sync
uv run python main.py
uv run python render_dishware_gallery.py --no-human --no-save
uv run pytest
```

Curated ACRONYM meshes/grasps ship under `data/acronym/` (default root). Set `ACRONYM_ROOT` only if you need a full external ACRONYM tree (e.g. re-curating the selection).

Canonical env construction:

```python
from table_bussing import DishwareCounts, make_env

env = make_env(DishwareCounts(plates=2, bowls=1), render_mode="human")
```
