"""Compare historical upload decode with orientation-correct disk decode."""
from pathlib import Path
import sys
import json
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'backend'))
from services.opponent_detection_service import detect_opponent_team_mobilenetv3_from_image
OUT = ROOT / 'backend/data/cv/reports/supplied_photos_20261019'
results = []
for path in sorted((OUT / 'inputs').iterdir()):
    with Image.open(path) as photo:
        orientation = photo.getexif().get(274)
        image = np.array(photo.convert('RGB'))[:,:,::-1].copy()
    result = detect_opponent_team_mobilenetv3_from_image(image, image_name=path.stem+'-raw-upload', save_debug=True)
    result.update(file=path.name, exifOrientation=orientation, decodedShape=list(image.shape))
    results.append(result)
    (OUT / 'raw_upload_predictions.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(path.name,orientation,[(s['name'],s['confidence']) for s in result['detectedTeam']],flush=True)
