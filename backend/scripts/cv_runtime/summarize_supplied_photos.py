"""Score manually reviewed species labels and build a before/after report."""
from pathlib import Path
import sys
import json
import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'backend'))
from services.pokemon_mobilenetv3_classifier_service import predict_pokemon_crops
OUT = ROOT / 'backend/data/cv/reports/supplied_photos_20261019'
corrected = json.loads((OUT/'predictions.json').read_text())
raw = {r['file']: r for r in json.loads((OUT/'raw_upload_predictions.json').read_text())}
expected = {}
for name in ['1000013389.jpg','1000013390.jpg','1000013391.jpg']:
    expected[name] = ['indeedee_female','gardevoir','annihilape','hydreigon','golisopod','pelipper']
for name in ['1000013393.jpg','1000013394.jpg','1000013396.jpg','1000013398.jpg']:
    expected[name] = ['pelipper','politoed','basculegion_female','golisopod','archaludon','venusaur']
for name in ['1000013404.jpg','1000013406(1).jpg','1000013406.jpg','1000013407.jpg']:
    expected[name] = ['goodra_hisui','incineroar','ninetales_alola','annihilape','venusaur','raichu']
expected['1000013402.jpg'] = ['maushold_family_of_four','staraptor','charizard','whimsicott','gardevoir','basculegion_male']
for name in ['1000013409.jpg','1000013410.jpg','1000013411.jpg']:
    expected[name] = ['charizard','rillaboom','floette_eternal','garchomp','basculegion_male','kingambit']
rows = []
for record in corrected:
    for slot, before, truth in zip(record['detectedTeam'],raw[record['file']]['detectedTeam'],expected[record['file']]):
        # Controlled wider crop from the SAME rectified card. Diagnostic only.
        card = np.array(Image.open(slot['debugCropPath']).convert('RGB'))[:,:,::-1].copy()
        h,w = card.shape[:2]
        wide = card[:, int(w*.18):int(w*.61)]
        wide_result = predict_pokemon_crops([wide])[0]
        wide_path = OUT / f"{Path(record['file']).stem}-wide-{slot['position']}.jpg"
        Image.fromarray(wide[:,:,::-1]).save(wide_path)
        rows.append(dict(image=record['file'],duplicate_of=record['duplicateOf'] or '',slot=slot['position'],expected=truth,
                         before=before['name'],before_confidence=before['confidence'],before_correct=before['name']==truth,
                         after=slot['name'],after_confidence=slot['confidence'],after_correct=slot['name']==truth,
                         wider=wide_result['pokemon'],wider_confidence=wide_result['confidence'],wider_correct=wide_result['pokemon']==truth,
                         crop=slot['debugPokemonCropPath'],wider_crop=str(wide_path)))
with (OUT/'scored_review.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
unique=[r for r in rows if not r['duplicate_of']]
metrics={k:dict(correct=sum(r[k+'_correct'] for r in unique),total=len(unique)) for k in ['before','after','wider']}
(OUT/'summary.json').write_text(json.dumps(dict(metrics=metrics,rows=rows),indent=2),encoding='utf-8')
print(metrics,flush=True)
page=Image.new('RGB',(1650,2200),'white');draw=ImageDraw.Draw(page)
font=ImageFont.truetype('arial.ttf',25);title=ImageFont.truetype('arial.ttf',38)
lines=[
    'Prediction and crop diagnosis - October 19, 2026',
    '15 supplied files; 14 unique photos; 84 unique Pokemon slots.',
    '1000013406(1).jpg and 1000013406.jpg are byte-identical.',
    '',
    'Confirmed primary cause: the upload decoder ignored EXIF rotation.',
    'All photos have orientation 6 (rotate 90 degrees clockwise).',
    'Stored pixels: 4000 x 3000. Correct display: 3000 x 4000.',
    'Disk-based OpenCV tests apply rotation; the old PIL upload path did not.',
    '',
    f"Old upload decode: {metrics['before']['correct']}/84 correct top-1 labels.",
    f"Rotation corrected: {metrics['after']['correct']}/84 correct top-1 labels.",
    f"Experimental wider crops: {metrics['wider']['correct']}/84 correct top-1 labels.",
    'Scores use manual visual labels; exclude the duplicate photo.',
    'Maushold exact family variant is provisional at this image resolution.',
    '',
    'Secondary issue: YOLO sometimes cuts off sprite parts.',
    'Raichu: missing tail / narrow body crop; Floette: missing flower parts.',
    'Wider crops are a diagnostic control, not a deployed crop change.',
    '',
    'Fix applied: EXIF transpose before converting upload pixels to BGR.',
    'Regression coverage: orientations 1, 3, 6 and 8; pixel and BGR checks.',
    'This report uses the active MobileNetV3 checkpoint with existing YOLO.',
    '',
    'Following pages: orientation-correct cards, actual crops and top five.',
    'Probabilities shown are model confidence, not calibrated accuracy.',
]
for i,line in enumerate(lines):
    draw.text((45,40+i*51),line,font=title if i==0 else font,fill='black')
draw.text((45,1450),'Old upload crops: sideways data sent to the cropper (1000013394.jpg)',font=font,fill='black')
for i,slot in enumerate(raw['1000013394.jpg']['detectedTeam']):
    crop=Image.open(slot['debugCropPath']).convert('RGB');crop.thumbnail((235,420))
    page.paste(crop,(45+i*263,1510))
    draw.text((45+i*263,1610),f"Slot {i+1}",font=font,fill='black')
page.save(OUT/'summary-review.png')
pages=[page]
evidence=Image.new('RGB',(1650,2200),'white');ed=ImageDraw.Draw(evidence)
ed.text((45,35),'Remaining crop failures after rotation correction',font=title,fill='black')
ed.text((45,95),'Actual classifier input versus experimental wider crop from the same card',font=font,fill='black')
for i,row in enumerate(r for r in unique if not r['after_correct']):
    y=180+i*470
    ed.text((45,y),f"{row['image']} / slot {row['slot']} / expected {row['expected']}",font=font,fill='black')
    for x,key in [(45,'crop'),(570,'wider_crop')]:
        crop=Image.open(row[key]).convert('RGB');crop.thumbnail((420,310))
        evidence.paste(crop,(x,y+50))
    ed.text((45,y+385),f"Actual: {row['after']} ({row['after_confidence']:.1%})",font=font,fill='black')
    ed.text((570,y+385),f"Wider: {row['wider']} ({row['wider_confidence']:.1%})",font=font,fill='black')
ed.text((45,2130),'Diagnostic only: wider crops also introduce one Maushold error (see CSV).',font=font,fill='black')
pages.append(evidence)
for r in corrected:
    image=Image.open(OUT/f"{Path(r['file']).stem}-review.jpg").convert('RGB')
    d=ImageDraw.Draw(image)
    d.rectangle((30,12,1175,95),fill='white')
    d.text((40,25),'Orientation corrected: YOLO crops + MobileNetV3',font=font,fill='black')
    d.text((40,65),r['file'],font=font,fill='black')
    pages.append(image)
pages[0].save(OUT/'prediction_crop_report.pdf',save_all=True,append_images=pages[1:],resolution=120)
