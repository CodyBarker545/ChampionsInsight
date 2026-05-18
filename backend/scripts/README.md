# Backend Script Layout

The scripts are grouped by purpose so related workflows live together:

- `data_build/`: reference generation, embedding indexes, and FAISS builders
- `cv_datasets/`: YOLO dataset builders, relabeling, and cleanup
- `cv_training/`: model training entrypoints
- `cv_runtime/`: inference and debug runners
- `review_tools/`: upload intake, crop harvesting, and clustering helpers
- `synthetic_data/`: synthetic image generation and audits
- `utilities/`: one-off inspection helpers

Top-level scripts are only orchestration or shell helpers:

- `build_all_data.py`
- `resume_synthetic_pokemon_batches.ps1`
- `run_selected_opponent_uploads.ps1`
