# Backend Data Layout

`backend/data/` is the home for repo-tracked source data, generated computer-vision assets, runtime uploads, and local user state used by the backend.

## Structure

- `pokemon/`
  - curated Pokemon battle data
  - sprite assets and metadata used by the app and CV workflows
- `competitive/`
  - `vgc_data/`: tournament CSV inputs
  - `analysis_json/`: generated tournament summaries and aggregates
  - `diagrams/`: generated charts and visual outputs
- `cv/`
  - `references/pokemon/`: manually curated Pokemon reference crops
  - `references/types/type_icons/`: single-type icon references
  - `references/types/type_combo_icons/`: generated dual-type references
  - `indexes/pokemon_embeddings/`: generated Pokemon embedding index files
  - `indexes/type_embeddings/`: generated type embedding index files
  - `debug/crops/`: saved crops and overlays from CV debugging
  - `debug/reports/`: generated CV JSON reports
- `uploads/`
  - uploaded opponent photos and latest detection artifacts used at runtime
- `user/`
  - local saved user team state
- `rag/docs/`
  - local fallback RAG documents stored under the data tree

## What Is Source Vs Generated

Mostly source or manually curated:

- `pokemon/`
- `competitive/vgc_data/`
- `cv/references/`

Mostly generated or runtime:

- `competitive/analysis_json/`
- `competitive/diagrams/`
- `cv/indexes/`
- `cv/debug/`
- `uploads/`
- `user/`

## Rebuild Paths

These folders are refreshed by `backend/scripts/build_all_data.py` and the scripts it orchestrates:

- `backend/vector_store/`
- `backend/data/cv/references/types/type_combo_icons/`
- `backend/data/cv/indexes/type_embeddings/`
- `backend/data/cv/indexes/pokemon_embeddings/`
- optionally `backend/data/cv/debug/reports/`

## Path Conventions

- backend code should import shared filesystem locations from `backend/paths.py`
- avoid hardcoding absolute paths in services or scripts
- treat `uploads/` and `user/` as local runtime state, not durable source-of-truth data
