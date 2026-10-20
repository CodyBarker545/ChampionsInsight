ChampionsInsight
AI-assisted Pokémon Champions battle preparation and tournament intelligence.

ChampionsInsight is a full-stack application for analyzing competitive Pokémon teams. It combines deep-learning computer vision to identify an opponent's Pokémon from a photo with tournament statistics, unsupervised team clustering, damage calculations, and retrieval-based game-mechanics help.

The aim is to turn a photo of a team-selection screen into actionable information: what Pokémon are present, which strategies might they represent, what sets appear in tournament data, and how might your own team match up?
Project status: Active development. Built primarily as a local, Windows-first application and an applied AI/software-engineering portfolio project. Outputs are decision-support estimates, not guaranteed predictions.

What can ChampionsInsight be used for?
For competitive players
- Identify opposing teams: Take or upload a photo of a team-selection screen to detect cards, locate sprites, and identify Pokémon.
- Prepare for matchups: View Pokémon details, level-50 stats, speed interactions, and calculated damage ranges.
- Investigate likely sets: Examine common tournament moves, items, and conditional set probabilities to inform your preparation.
- Understand team styles: Explore rule-defined strategies such as Rain or Trick Room, alongside groups of similar tournament teams discovered through clustering.
- Research the metagame: Explore tournament usage, common team compositions, moves, items, and competitive trends.
- Review mechanics: Ask questions about status effects, items, and other mechanics using locally indexed reference documents.
- Build teams: Save and inspect your own roster and browse recommended builds derived from available tournament data.
For developers and researchers
ChampionsInsight is also a practical example of an end-to-end ML-enabled application: image acquisition, object detection, classification, model-confidence handling, data transformation, unsupervised learning, vector search, API integration, frontend presentation, and automated tests.
How it works
1. Computer vision: photo to Pokémon identities
Camera photo or uploaded image
              |
              v
       OpenCV / NumPy
  preprocessing and quality checks
              |
              v
     YOLOv8n card detector
     locate opponent team cards
              |
              v
      Second YOLO detector
       locate Pokémon sprites
              |
              v
       MobileNetV3 classifier
   rank candidate Pokémon species
              |
              v
    Confidence checks + recovery
              |
              v
     Identified Pokémon + stats
The active Pokémon identity pipeline uses two stages of YOLO object detection and MobileNetV3 image classification. It includes confidence-based checks (with an approximately 85% acceptance threshold in the current workflow), higher-resolution retries for missed cards, and wider-crop retries for uncertain classifications. Detection results can be flagged for review rather than treated as certain.

Earlier DINOv2/FAISS image-embedding experiments remain in the codebase for research and comparison, but are not the active application identity predictor.

Important: Recognition accuracy depends on the camera angle, screen glare, crop quality, sprite appearance, and training-data coverage. Identifying all six Pokémon reliably remains an area of ongoing improvement.
2. Tournament intelligence: data analysis and machine learning
Tournament records feed several distinct kinds of analysis:
Capability	Method	What it provides
Pokémon usage, moves, items, and trends	Descriptive statistical analysis	Evidence of what appears in recorded tournaments
Likely sets and conditional probabilities	Frequency-based statistical estimates	Observed distributions given available conditions
Rain, Trick Room, and other known styles	Handwritten rules	Interpretable strategy labels
Similar tournament team discovery	K-means clustering	Data-driven team groupings and representative examples
Mechanics document retrieval	Text embeddings + FAISS, with document fallback	Relevant reference passages for user questions


The team-clustering implementation uses engineered team features, TF-IDF weighting, feature standardization, TruncatedSVD, and scikit-learn KMeans. Cluster summaries help expose common Pokémon, moves, items, abilities, and representative teams.

Scope clarification: Most metagame summaries are statistical analysis, not trained predictive models. ChampionsInsight does not currently train a model to predict future metagames or battle winners.
3. Battle and knowledge tools
- Damage calculations: Integrates the bundled @smogon/calc calculation service to evaluate battle scenarios.
- Team and Pokédex views: Browse Pokémon information, builds, and relevant competitive summaries.
- Mechanics Q&A: Retrieves information from curated local documents via embedding-based search or local-document fallback. Current answers are primarily document-grounded/extractive, not an always-on generative LLM.
Technology stack
Layer	Technologies
Frontend	React, JavaScript, Vite, CSS
API	Python, Flask, Flask-CORS
Deep learning	PyTorch, Ultralytics YOLO, MobileNetV3
Image processing	OpenCV, NumPy, Pillow
Data analysis	pandas, NumPy
Unsupervised learning	scikit-learn (K-means, TF-IDF, TruncatedSVD, scaling)
Retrieval	Sentence Transformers, FAISS, curated local reference documents
Battle calculations	@smogon/calc (Node.js service)
Testing	pytest, Vitest, React Testing Library


Project structure
ChampionsInsight/
├── backend/
│   ├── api/               # Flask routes
│   ├── services/          # CV, tournament, battle, and other application logic
│   ├── rag/               # Document processing and retrieval
│   ├── scripts/           # Data builds, ML/CV workflows, training, and reports
│   ├── data/              # Pokémon, tournament, CV and derived datasets
│   ├── app.py             # Backend application entry point
│   ├── requirements.txt
│   └── requirements-dev.txt
├── frontend/
│   ├── src/               # React pages, components, API clients, tests
│   └── package.json
├── scripts/
│   └── setup.ps1          # Windows setup and data bootstrap
├── docs/
└── README.md
Some model weights, generated indexes, datasets, and training artifacts may be large, generated locally, or excluded from Git. A clean checkout may require building data or supplying model artifacts before all features work.
Getting started (Windows / PowerShell)
Prerequisites
- Python 3 with venv
- Node.js and npm
- PowerShell
Node.js is used by both the React frontend and the bundled Smogon damage-calculation process.
Option A: Automated setup
From the repository root:
git clone https://github.com/CodyBarker545/ChampionsInsight.git
cd ChampionsInsight
.\scripts\setup.ps1
If PowerShell blocks the script:
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
The setup script creates the Python virtual environment, installs backend and frontend dependencies, and builds derived data.

Available setup options:
.\scripts\setup.ps1 -SkipFrontend
.\scripts\setup.ps1 -SkipData
.\scripts\setup.ps1 -WithDebugReport
Building data can take time and may depend on datasets and artifacts available in your checkout. -SkipData skips those build steps when you want to install dependencies first.

Option B: Manual setup
From the repository root:
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r backend\requirements.txt

cd frontend
npm install
cd ..
Build derived data:
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py
Useful build options:
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py --dry-run
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py --skip-rag
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py --skip-cv
.\.venv\Scripts\python.exe backend\scripts\build_all_data.py --with-debug-report
Legacy image-embedding index builders are optional offline experiments (--with-legacy-cv).
Run the application
Terminal 1 — Flask backend (from repository root):
.\.venv\Scripts\python.exe backend\app.py
Backend: https://localhost:5000

Terminal 2 — React frontend:
cd frontend
npm run dev
Frontend: https://127.0.0.1:5173

Vite uses local HTTPS and proxies /api requests to the Flask backend. Your browser may ask you to trust the development certificate.
Use the camera from a phone
With the phone and development computer on the same network, open:
https://YOUR_COMPUTER_IPV4:5173/camera
Find the development computer's IPv4 address with ipconfig. The local development certificate may need to be accepted on the phone.
Key backend capabilities
Area	Purpose
Opponent detection	Image upload, quality assessment, YOLO/MobileNet prediction, result retrieval
Tournament intelligence	Archetype summaries, clustering, team classification, usage trends
Set probabilities	Common sets and conditional tournament frequencies
Battle analysis	Damage and matchup calculation integration
Pokémon/Pokédex	Pokémon information, stats, recommended builds
Local mechanics Q&A	Document retrieval and source-grounded answers
User team	Local team creation, saving, and review


Relevant API routes include:
POST /api/opponent/image
POST /api/opponent/detect-mobilenetv3
GET  /api/opponent/prediction/latest
GET  /api/tournament/archetypes/rules
GET  /api/tournament/archetypes/clusters
POST /api/tournament/archetypes/classify
POST /api/rag/ask
The legacy /api/opponent/detect embedding-based route is disabled in favor of MobileNetV3.
Testing and validation
Run backend tests from the repository root:
.\.venv\Scripts\python.exe -m pytest
Run frontend tests:
cd frontend
npm test
Check the production frontend build:
npm run build
For model assessment, useful evaluation artifacts include YOLO precision/recall/mAP, MobileNet accuracy and macro-F1, confusion matrices, and end-to-end six-Pokémon detection success on unseen real camera photos. Do not assume a classifier's validation accuracy is equal to complete pipeline accuracy.
Current limitations
- The image-recognition workflow is still being improved for glare, unusual framing, incomplete card detections, and uncertain Pokémon crops.
- Tournament statistics reflect the coverage and quality of the collected dataset; observed frequencies are not certainty about a particular opponent.
- K-means clusters identify similarity groups, but their strategic interpretation may require manual labeling or review.
- The app is primarily designed for local development, not yet documented as a production-hosted service.
- The current mechanics helper performs grounded retrieval/extraction; LLM-generated tactical advice is not presented as a fully implemented feature.
Potential future work
These are possible extensions, not claims about features already shipped:
- More reliable layout-aware recovery of all six team cards.
- A public, hosted demo with streamlined setup.
- Confidence calibration and comprehensive real-photo evaluation reports.
- Team recommendation and counter-team ranking evaluated on held-out tournament data.
- Time-aware metagame forecasting and trend-change detection.
- Retrieval-augmented, source-cited battle advice using an optional language model.
- Additional monitoring, model-versioning, and deployment automation.
Why this project exists
ChampionsInsight explores how computer vision, unsupervised learning, and statistical analysis can operate together inside a real application. It is designed both as a useful competitive preparation tool and as an applied software-engineering/AI portfolio project involving:
- Dataset preparation and model integration.
- Multi-stage computer vision with fallbacks and confidence handling.
- Clustering and feature engineering from structured tournament records.
- Vector-based information retrieval.
- Python API development and React user interfaces.
- Testing, reproducibility, error analysis, and iterative improvement.
Disclaimer
ChampionsInsight is an independent fan-created project and is not affiliated with, endorsed by, or sponsored by Nintendo, The Pokémon Company, or GAME FREAK. Pokémon names, characters, and related marks belong to their respective owners.

Competitive recommendations and predicted identities are informational estimates and should be verified where accuracy matters.
