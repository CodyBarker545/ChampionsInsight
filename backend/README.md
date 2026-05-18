# Champions Insight Backend

Flask API for battle analysis, Pokedex data, RAG answers, user team storage, and opponent image detection.

## Setup

From the repo root, prefer the full setup command:

```powershell
.\scripts\setup.ps1
```

To rebuild backend-derived data only:

```powershell
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py
```

## Run

```powershell
cd D:\SoftwareEng\ChampionsInsight
.\.venv\Scripts\python.exe backend\app.py
```

The API runs at `https://localhost:5000`.

## Derived Data

`backend/scripts/build_all_data.py` rebuilds:

- RAG vector index and chunk metadata in `backend/vector_store/`
- generated type-combo reference images and metadata
- type icon embedding index
- Pokemon sprite embedding index
- pruned Pokemon embedding metadata

Use `--dry-run` to print the exact steps without running them.

## Script Layout

- `data_build/`: reference generation and embedding/index builders
- `cv_datasets/`: YOLO dataset creation and cleanup
- `cv_training/`: model training entrypoints
- `cv_runtime/`: inference/debug runners
- `review_tools/`: upload intake, crop harvesting, and clustering helpers
- `synthetic_data/`: synthetic image generation and audits
- `utilities/`: one-off inspection helpers
