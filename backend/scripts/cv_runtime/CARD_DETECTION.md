# Validated opponent card detection

The active upload path detects full cards with the existing
`yolov8n_slots_mixed_synth_real_cv/weights/best.pt` checkpoint. It retains the
existing inner-object YOLO, MobileNetV3 checkpoint, and Flask routes. No models
were retrained. The old fixed guide coordinates are not used for cropping.

`services/card_detection_service.py` owns full-card geometry:

- Initial YOLO inference: confidence 0.30, image size 960.
- If fewer than six validated cards remain: confidence 0.12, image size 1536.
- Image-bounded boxes, shape/red-card evidence validation, IoU/containment NMS,
  and rejection of boxes overlapping another accepted card by more than 10%
  of the smaller card area.
- Recovery searches image-backed card bands in columns anchored to detected
  card geometry. Local dimensions and center alignment permit scale changes
  and moderate perspective. Recovery extends the observed layout in either
  direction, but never accepts an inferred position without visible evidence.
- Cards are ordered vertically. Internal gaps are retained using observed
  spacing. With incomplete teams, leading/trailing placement can be ambiguous;
  `positionsAreRelative` explicitly says numbering starts at the first visible
  card. No absolute missing leading-card position is asserted without evidence.

The response retains `detectedTeam` and adds `cardDetection`,
`unresolvedPositions`, and `positionsAreRelative`. Each accepted card includes
its position, confidence, source, and validation state. `cardConfidence` and
`cardSource` accompany the MobileNet prediction. Unresolved cards are absent
from the backend detection array. The battle-prep UI retains six editable
slots, places validated cards by position, and shows unresolved-position
guidance. Existing replacement search and manual selection still work.

When debug is enabled, each input receives `card-detections.jpg` with box,
position, source, and score annotations; individual card and sprite images are
saved alongside it. Layout evidence scores are red support, not calibrated
YOLO probabilities. Use the source field to distinguish them.

## Tests (from repository root, PowerShell)

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_card_detection.py backend/tests/test_pokemon_crop_recovery.py backend/tests/test_cv_service.py backend/tests/test_opponent_detection_workflow.py backend/tests/test_routes.py -q
npm --prefix frontend test -- --run src/pages/BattlePrepPage.test.jsx
npm --prefix frontend run build
```

Unit cases cover zero through six detections, low-confidence retries, valid-box
preservation, duplicate/contained boxes, shifted framing, scale, trapezoidal
cards, internal missing cards, partial edge fragments, bounds, overlap,
bidirectional recovery, saved debug images, and zero-card classification.
API tests check partial response persistence; UI tests check editable gaps.

## Reproduce the 15-photo comparison

```powershell
.\.venv\Scripts\python.exe backend/scripts/cv_runtime/compare_card_detection.py --baseline backend/data/cv/reports/card_detection_layout_20261019/baseline.json --output backend/data/cv/reports/card_detection_layout_rerun
```

The baseline manifest records exact original paths and manual visual labels.
Keep those original files available or update only their `source` paths. The
script uses the actual upload decoder, writes `after.json`, `comparison.json`,
`comparison.md`, full annotated images, individual crops, and per-photo review
images. It does not write predictions back into the app's latest-upload state.

On these 15 full-resolution development photos:

| Measure | Before | After |
| --- | ---: | ---: |
| Six actual detected cards | 6/15 (40%) | 15/15 (100%) |
| Correct accepted identities | 70/90 | 89/90 |
| Incorrect accepted identities | 4/90 | 0/90 |
| Identity review required | 16/90 | 1/90 |

Before counts refer to the old red-card detector, before its fixed-coordinate
fallback fabricated six crops. The remaining identity review is Rillaboom in
`1000013417.jpg` (84.65%, below the unchanged 85% acceptance threshold).
These are development results, not a held-out generalization benchmark.
