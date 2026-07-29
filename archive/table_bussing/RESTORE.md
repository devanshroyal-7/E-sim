# Restoring table bussing

Archived on 2026-07-29 when the project focus moved away from `TableBussing-v1`.
Layout inside this folder mirrors the old repo-root layout so a restore is a straight move back.

## Restore to repo root

From the repository root:

```bash
# Modules
mv archive/table_bussing/table_bussing.py .
mv archive/table_bussing/environment.py .
mv archive/table_bussing/acronym_dishware.py .
mv archive/table_bussing/dishware_actors.py .
mv archive/table_bussing/render_dishware_gallery.py .

# Package + data + tests + script
mv archive/table_bussing/envs .
mv archive/table_bussing/data .
mv archive/table_bussing/tests/* tests/
mv archive/table_bussing/scripts/settable_test.py scripts/

# Optional: drop the empty archive shell
rmdir archive/table_bussing/scripts archive/table_bussing/tests 2>/dev/null
rm -rf archive/table_bussing
```

Then restore the old README / `pyproject.toml` description from git history if you want the docs to match again:

```bash
git log --oneline -- README.md pyproject.toml
```

## Quick check after restore

```bash
uv sync
uv run python environment.py          # interactive TableBussing smoke
uv run python render_dishware_gallery.py --no-human --no-save
uv run pytest
```

## Contents

| Path | Role |
|------|------|
| `table_bussing.py` | `TableBussing-v1` env, `DishwareCounts`, `make_env` |
| `environment.py` | Thin re-export + interactive smoke demo |
| `acronym_dishware.py` | ACRONYM catalog / curated selection / mesh helpers |
| `dishware_actors.py` | Shared Sapien actor construction |
| `render_dishware_gallery.py` | Visual QA gallery env |
| `envs/` | Package re-exports |
| `data/` | Curated ACRONYM meshes/grasps + selection JSON |
| `tests/` | Unit + integration tests (and ACRONYM `conftest`) |
| `scripts/settable_test.py` | Legacy shim for an older test entry |

Regenerated artifacts that may be present under `data/` (often gitignored):

- `acronym_dishware_manifest.json`
- `dishware_gallery.png`
