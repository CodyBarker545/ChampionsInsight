# Backend Script Layout

`backend/scripts/` contains the one-off and repeatable workflows used to build data, inspect computer-vision outputs, prepare datasets, and train or validate supporting models.

## Top-Level Entry Points

- `build_all_data.py`
  - orchestrates the standard derived-data rebuild used by fresh setup
  - can skip RAG, skip CV, run dry, or include the uploaded-opponent debug report
- `resume_synthetic_pokemon_batches.ps1`
  - PowerShell helper for continuing synthetic Pokemon generation batches
- `run_selected_opponent_uploads.ps1`
  - PowerShell helper for targeted upload-review workflows

## Directory Groups

- `data_build/`
  - build scripts for RAG artifacts, type-combo references, and embedding indexes
- `cv_runtime/`
  - runtime checks, upload debug runs, and detection validation helpers
- `cv_datasets/`
  - dataset creation, relabeling, cleanup, and export prep
- `cv_training/`
  - training entrypoints and related model workflows
- `review_tools/`
  - upload intake, crop harvesting, clustering, and manual review helpers
- `synthetic_data/`
  - synthetic image generation, pruning, and audit scripts
- `utilities/`
  - ad hoc inspection and maintenance utilities

## Common Workflow

For a normal local rebuild from the repo root:

```powershell
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py
```

Useful variants:

```powershell
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py --dry-run
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py --skip-rag
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py --skip-cv
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py --with-debug-report
```

## Expectations

- run Python scripts from the repo root unless a script says otherwise
- prefer shared path helpers from `backend/paths.py` when adding new scripts
- keep generated outputs in the established data folders instead of inventing new scratch locations
