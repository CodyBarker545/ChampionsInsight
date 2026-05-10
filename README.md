# Champions Insight

Champions Insight is a local battle-prep app for Pokemon Champions-style team review. It combines:

- a React frontend with team builder, battle prep, Pokedex, RAG help, and guided opponent camera capture
- a Flask backend with battle analysis, user team storage, Pokedex data, tournament usage summaries, and opponent image detection
- computer-vision pipelines for opponent card quality checks, type icon detection, Pokemon sprite matching, and debug reports
- local RAG over curated Pokemon mechanics docs

## Project Layout

```text
backend/
  api/        Flask route blueprints
  data/       Pokemon, competitive, CV, upload, and user data
  rag/        RAG preprocessing, vector indexing, and answer helpers
  scripts/    Reproducible data/index build and CV debug scripts
  services/   Backend business logic
frontend/
  src/        React app, pages, components, API helpers, styles
scripts/
  setup.ps1   Fresh-machine setup command
docs/         Planning docs and mockups
```

## Fresh Machine Setup

From a new clone on Windows, run:

```powershell
cd D:\SoftwareEng\ChampionsInsight
.\scripts\setup.ps1
```

If PowerShell blocks local scripts on that machine, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

That command:

- creates `.venv`
- installs backend Python requirements
- installs frontend npm packages
- builds the RAG vector index
- rebuilds generated type-combo references
- builds type icon and Pokemon sprite embedding indexes
- prunes blocked Pokemon embedding references

The first run can take a while because PyTorch, TorchVision, SentenceTransformers, and model weights may need to download.

Optional setup flags:

```powershell
.\scripts\setup.ps1 -SkipFrontend
.\scripts\setup.ps1 -SkipData
.\scripts\setup.ps1 -WithDebugReport
```

`-WithDebugReport` also runs the upload debug report against whatever photos are currently in `backend\data\uploads`.

## Rebuild Data Only

After dependencies are installed, rebuild all derived data artifacts with:

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

The data builder runs these steps in order:

1. `python -m rag.preprocess`
2. `python -m rag.build_faiss_index`
3. `python scripts/build_type_combo_references.py`
4. `python scripts/build_type_embedding_index.py`
5. `python scripts/build_pokemon_embedding_index.py`
6. `python scripts/prune_blocked_embedding_index.py`

Generated outputs include:

- `backend/vector_store/pokemon_rag_embeddings.npy`
- `backend/vector_store/chunks.json`
- `backend/data/cv/references/types/type_combo_icons/type_combo_metadata.json`
- `backend/data/cv/indexes/type_embeddings/`
- `backend/data/cv/indexes/pokemon_embeddings/`

Source data expected in the repo:

- `backend/RAGDocs/*.txt`
- `backend/data/pokemon/`
- `backend/data/competitive/`
- `backend/data/cv/references/`

## Run The App

Start the backend:

```powershell
cd D:\SoftwareEng\ChampionsInsight
.\.venv\Scripts\python.exe backend\app.py
```

Backend URL:

```text
https://localhost:5000
```

Start the frontend:

```powershell
cd D:\SoftwareEng\ChampionsInsight\frontend
npm run dev
```

Frontend URL:

```text
https://127.0.0.1:5173
```

Vite uses a local HTTPS dev certificate through `@vitejs/plugin-basic-ssl`. The guided camera requires HTTPS.

## Use From A Phone

Make sure the phone and computer are on the same Wi-Fi network.

Start backend and frontend as above, then find the computer IPv4 address:

```powershell
ipconfig
```

Open this on the phone:

```text
https://YOUR_IPV4_ADDRESS:5173
```

Accept the local development certificate warning if prompted. The guided camera overlay is tuned from successful captures in `backend/data/uploads`: the six red opponent slots should fill nearly the full vertical frame, and the type icons should sit inside the right-side gold target.

## Main Backend Endpoints

- `GET /api/health`
- user team load/save routes
- battle matchup and calculator routes
- Pokemon search, detail, stats, and moves routes
- Pokedex grid/detail routes
- RAG question route
- opponent upload, quality check, detection, and latest prediction routes

## Tests

Backend:

```powershell
cd D:\SoftwareEng\ChampionsInsight
.\.venv\Scripts\python.exe -m pytest
```

Frontend:

```powershell
cd D:\SoftwareEng\ChampionsInsight\frontend
npm test
```

Build frontend:

```powershell
cd D:\SoftwareEng\ChampionsInsight\frontend
npm run build
```
