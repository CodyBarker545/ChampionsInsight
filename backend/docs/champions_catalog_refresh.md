# Refresh Champions data and training crops

From the repository root, compare the complete normal and shiny archive categories:

```powershell
.\.venv\Scripts\python.exe backend/scripts/data_build/sync_champions_catalog.py
```

Add `--apply` to download missing sprites and update the JSON/CSV sprite catalogs,
Pokédex form lookup, held-item dictionary, Champions item suggestions, and item
reference documents. The sync follows MediaWiki continuation across every page.
It preserves `Menu_CP_` filenames despite the archive's rename to `Menu Champs`.
Existing source files are retained. An unresolved form stops the update for review.

The last nonempty update is saved in
`backend/data/pokemon/champions_catalog_update.json`. Generate only its new classes:

```powershell
.\.venv\Scripts\python.exe backend/scripts/synthetic_data/prepare_champions_training_update.py --target-per-class 250 --workers 4
```

Each new non-Mega class gets **250 total images: 125 normal and 125 shiny**.
Both variants share a class folder in
`backend/data/training_dataset/slot_pokemon_synthetic/`. Mega sprites remain
available in the app but are excluded from slot-classifier training, matching
the existing workflow. The generator retains the existing red-card backgrounds,
crop visibility checks, blur, glare, noise, and JPEG augmentation.

Generation resumes without overwriting existing images, including when earlier
images were pruned. `--trim-excess` explicitly trims only synthetic files belonging
to this update's new source sprites. It does not prune the older dataset.
Counts are verified and written to
`backend/data/training_dataset/champions_training_update_summary.json`.

Restart the backend to reload cached catalogs. Item suggestions are served by
`GET /api/items`; availability describes the game catalog, not ranked legality.

The saved May 22 MobileNet run sampled 65,756 images across 264 classes, capped
at 250 per class (46,029 train / 9,863 validation / 9,864 test). To train a new
checkpoint with the same class cap:

```powershell
.\.venv\Scripts\python.exe backend/scripts/train_mobilenetv3_pokemon_classifier.py --target-images-per-class 250 --output-dir backend/data/cv/models/pokemon_mobilenetv3_updated
```

For an incremental roster update, start from the current checkpoint. Matching
classes keep their trained output weights even when their indices change;
new classes receive new output rows. For example:

```powershell
.\.venv\Scripts\python.exe backend/scripts/train_mobilenetv3_pokemon_classifier.py --target-images-per-class 250 --init-checkpoint backend/data/cv/models/pokemon_mobilenetv3/pokemon_mobilenetv3.pt --output-dir backend/data/cv/models/pokemon_mobilenetv3_updated --batch-size 64 --learning-rate 0.0001 --num-workers 2 --disable-amp --epochs 20 --patience 4
```

Each validation improvement saves `best_checkpoint.pt` and class mappings. The
training history is saved after every epoch. Final evaluation and report files
are written after training ends.

The GTX 1660 Ti was faster with standard contiguous tensors and AMP disabled in
a local benchmark. `--channels-last` and `--persistent-workers` are optional;
enable them only after checking speed and available RAM on the training machine.

This writes a separate model for evaluation. Dataset generation does not retrain
the active checkpoint or rebuild embedding/RAG indexes. Existing inference
models must be retrained or their indexes rebuilt to recognize the new classes.
