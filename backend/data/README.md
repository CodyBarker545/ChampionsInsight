# Backend Data Layout

This folder is the single home for backend data, runtime files, generated
computer-vision artifacts, and reference assets.

## Structure

- `pokemon/`
  - Curated Pokemon battle data.
  - Champion sprite metadata and sprite images.
- `competitive/`
  - `vgc_data/`: raw tournament CSV exports.
  - `analysis_json/`: generated usage summaries, top moves, and top items.
  - `diagrams/`: generated charts from competitive analysis scripts.
- `cv/`
  - `references/pokemon/`: custom camera/reference Pokemon crops.
  - `references/types/type_icons/`: single type icon references.
  - `references/types/type_combo_icons/`: generated real dual-type combo references.
  - `indexes/pokemon_embeddings/`: generated Pokemon embedding index.
  - `indexes/type_embeddings/`: generated type icon embedding index.
  - `debug/crops/`: generated CV crop/overlay debug images.
  - `debug/reports/`: generated CV JSON reports.
- `uploads/`
  - Runtime opponent photo uploads and latest prediction JSON.
- `user/`
  - Local saved user team data.
- `rag/docs/`
  - Local fallback RAG documents.

Application code should import paths from `backend/paths.py` instead of
hardcoding folder names.
