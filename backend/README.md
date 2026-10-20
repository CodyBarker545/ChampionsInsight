# Champions Insight Backend

The backend is a Flask application that serves the local API, starts the bundled Smogon damage calculator helper, loads tournament and Pokemon data, answers RAG questions, and runs the opponent-detection pipeline.

## Backend Layout

- `app.py`: Flask entrypoint and service warmup
- `api/`: route blueprints grouped by domain
- `services/`: business logic used by the routes
- `data/`: source data, runtime uploads, generated CV artifacts, and saved user data
- `rag/`: preprocessing and FAISS indexing for mechanics documents
- `scripts/`: rebuild and support workflows
- `tests/`: backend pytest suite and fixtures

## Setup

From the repo root, the recommended bootstrap remains:

```powershell
.\scripts\setup.ps1
```

Manual install:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r backend\requirements.txt
cd frontend
npm install
```

`npm install` is still required even when you only care about the backend, because `backend/app.py` starts the bundled Node-based damage calculator helper.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe backend\app.py
```

The backend serves HTTPS on:

```text
https://localhost:5000
```

What starts with it:

- Flask API on port `5000`
- background warmup for the active YOLO/MobileNet camera models, then question answering
- bundled Smogon damage calculator process on `http://127.0.0.1:3001`

## API Surface

The main route groups are:

- `health_routes`
- `user_team_routes`
- `battle_routes`
- `pokemon_routes`
- `pokedex_routes`
- `rag_routes`
- `opponent_routes`
- `tournament_routes`
- `set_probability_routes`

## Derived Data

`backend/scripts/build_all_data.py` is the backend's main rebuild entrypoint.

It can rebuild:

- RAG chunk metadata and vector embeddings
- tournament usage, moves, items, set probabilities, archetypes, meta intelligence, and recommended builds
- optional uploaded-opponent debug reports

Legacy CV reference/index builders are offline-only; enable them with `--with-legacy-cv`.

Common commands:

```powershell
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py --dry-run
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py --skip-rag
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py --skip-cv
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py --with-debug-report
```

## Test And Debug

Run the backend test suite from the repo root:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Useful areas under `backend/scripts/`:

- `data_build/` for rebuild steps
- `cv_runtime/` for runtime detection checks and reports
- `review_tools/` for dataset review helpers
- `cv_training/` and `synthetic_data/` for model and dataset work

### Main-app dependencies

`backend/requirements.txt` installs the app runtime. For tests, catalog scraping,
and model-training reports, install `backend/requirements-dev.txt` instead.
The app uses YOLO + MobileNetV3 for photo prediction. Type embedding experiments
remain available offline and are excluded from the main app startup and release
artifact requirements. Training data, reports, and
tournament backups remain in the source workspace.
