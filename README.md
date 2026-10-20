# Champions Insight

Champions Insight is a local Pokemon Champions battle-prep app. It combines a React frontend, a Flask API, a bundled Smogon damage calculator backend, local RAG over curated mechanics notes, tournament usage analysis, and computer-vision workflows for opponent team capture.

## What The App Does

- build and save your own team locally
- review matchup damage and speed interactions
- inspect Pokedex details, common moves, and recommended teams
- ask mechanics questions through the local RAG helper
- estimate likely opponent sets from tournament data
- classify team archetypes from tournament trends
- capture or upload opponent photos and detect the opposing team

## Project Layout

```text
backend/
  api/        Flask route blueprints
  data/       Source data, generated CV assets, uploads, and local user data
  rag/        RAG preprocessing and FAISS index builders
  scripts/    Data-build, CV, training, review, and utility workflows
  services/   Backend business logic
frontend/
  src/        React app, pages, API clients, tests, and styles
scripts/
  setup.ps1   Windows bootstrap script for dependencies and derived data
docs/         Product docs, mockups, and planning notes
```

## Prerequisites

Champions Insight is currently set up as a Windows-first local project.

- Python 3 with `venv`
- Node.js and `npm`
- PowerShell

Node is required for two reasons:

- the Vite frontend
- the bundled Smogon damage calculator process that the Flask app starts automatically on port `3001`

## Recommended Setup

From the repository root:

```powershell
cd D:\SoftwareEng\ChampionsInsight
.\scripts\setup.ps1
```

If local script execution is blocked:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

The setup script will:

- create `.venv` if it does not exist
- upgrade `pip`
- install `backend/requirements.txt`
- run `npm install` in `frontend/`
- build derived RAG and computer-vision data unless skipped

Optional flags:

```powershell
.\scripts\setup.ps1 -SkipFrontend
.\scripts\setup.ps1 -SkipData
.\scripts\setup.ps1 -WithDebugReport
```

`-WithDebugReport` also runs the uploaded-opponent detection report against the current files in `backend/data/uploads/`.

## Manual Setup

If you want to do the steps yourself:

```powershell
cd D:\SoftwareEng\ChampionsInsight
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r backend\requirements.txt
cd frontend
npm install
```

Then build derived data:

```powershell
cd D:\SoftwareEng\ChampionsInsight
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py
```

## Derived Data Build

`backend/scripts/build_all_data.py` rebuilds the generated assets a fresh install needs:

- RAG chunk previews and FAISS embeddings in `backend/vector_store/`
- tournament usage, moves, items, set probabilities, archetypes, meta intelligence, and recommended builds
- optional uploaded-opponent debug reports in `backend/data/cv/debug/reports/`

Legacy CV reference/index builders are offline-only; enable them with `--with-legacy-cv`.

Useful variants:

```powershell
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py --dry-run
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py --skip-rag
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py --skip-cv
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py --with-debug-report
```

## Running The App

Start the backend from the repo root:

```powershell
cd D:\SoftwareEng\ChampionsInsight
.\.venv\Scripts\python.exe backend\app.py
```

Backend URL:

```text
https://localhost:5000
```

Start the frontend in a second terminal:

```powershell
cd D:\SoftwareEng\ChampionsInsight\frontend
npm run dev
```

Frontend URL:

```text
https://127.0.0.1:5173
```

Notes:

- Vite serves HTTPS with `@vitejs/plugin-basic-ssl`.
- `/api` requests are proxied to `https://localhost:5000`.
- the camera flow needs HTTPS, which is why local dev runs over TLS.

## Using The Camera From A Phone

Make sure the phone and dev machine are on the same network, then open:

```text
https://YOUR_IPV4_ADDRESS:5173/camera
```

You can find the computer's IPv4 address with:

```powershell
ipconfig
```

Expect to accept the local certificate warning on the phone.

## Main Backend Areas

- `health_routes`: backend health checks
- `user_team_routes`: load and save the local user team
- `battle_routes`: matchup analysis and damage-calculator bridge
- `pokemon_routes`: Pokemon search, stats, moves, and meta intelligence
- `pokedex_routes`: Pokedex grid, details, and recommended teams
- `rag_routes`: mechanics Q&A over local docs
- `opponent_routes`: upload, quality-check, and opponent detection flows
- `tournament_routes`: tournament usage and analysis data
- `set_probability_routes`: likely set estimation for detected teams

## Testing

Backend tests:

```powershell
cd D:\SoftwareEng\ChampionsInsight
.\.venv\Scripts\python.exe -m pytest
```

Frontend tests:

```powershell
cd D:\SoftwareEng\ChampionsInsight\frontend
npm test
```

Frontend production build:

```powershell
cd D:\SoftwareEng\ChampionsInsight\frontend
npm run build
```

### Main-app dependencies

`backend/requirements.txt` installs the app runtime. For tests, catalog scraping,
and model-training reports, install `backend/requirements-dev.txt` instead.
The app uses YOLO + MobileNetV3 for photo prediction. Type embedding experiments
remain available offline and are excluded from the main app startup and release
artifact requirements. Training data, reports, and
tournament backups remain in the source workspace.
