"""Reproduce live upload predictions and save reviewable crops for supplied photos."""
from pathlib import Path
import sys
import json
import hashlib
import zipfile
import shutil
import time
import argparse

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'backend'))
from PIL import Image, ImageOps, ImageDraw, ImageFont
from services import cv_service
from services.opponent_detection_service import detect_opponent_team_mobilenetv3_from_image

OUT = ROOT / 'backend/data/cv/reports/supplied_photos_20261019'

def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=OUT)
    OUT = parser.parse_args().output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = OUT / 'inputs'
    inputs.mkdir(exist_ok=True)
    names = ['1000013394.jpg','1000013396.jpg','1000013390.jpg','1000013406(1).jpg','1000013407.jpg','1000013406.jpg','1000013393.jpg','1000013398.jpg','1000013402.jpg','1000013404.jpg','1000013389.jpg','1000013391.jpg']
    for name in names:
        shutil.copy2(Path('D:/Downloads') / name, inputs / name)
    with zipfile.ZipFile('D:/Downloads/attachments(2).zip') as archive:
        for member in archive.infolist():
            if Path(member.filename).suffix.lower() in {'.jpg','.jpeg','.png'}:
                (inputs / Path(member.filename).name).write_bytes(archive.read(member))
    results = []
    pages = []
    hashes = {}
    font = ImageFont.truetype('arial.ttf', 23)
    small = ImageFont.truetype('arial.ttf', 19)
    for path in sorted(inputs.iterdir()):
        print('Processing', path.name, flush=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        start = time.time()
        result = detect_opponent_team_mobilenetv3_from_image(
            cv_service.read_cv_image(path),
            image_name=f'{OUT.name}-{path.name}',
            save_debug=True,
        )
        result.update(file=path.name, sha256=digest, duplicateOf=hashes.get(digest), elapsedSeconds=round(time.time()-start, 2))
        hashes.setdefault(digest, path.name)
        results.append(result)
        (OUT / 'predictions.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        page = Image.new('RGB', (1650,2200), 'white')
        draw = ImageDraw.Draw(page)
        draw.text((40,25), 'Upload pipeline: YOLO + MobileNetV3 with crop recovery', font=font, fill='black')
        draw.text((40,65), path.name + (f" (duplicate of {result['duplicateOf']})" if result['duplicateOf'] else ''),font=small,fill='black')
        original = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        original.thumbnail((400,270))
        page.paste(original,(1200,100))
        draw.text((40,110), 'Each row: detected card / classifier input / top five predictions',font=small,fill='black')
        for slot in result['detectedTeam']:
            y = 400 + (slot['position']-1)*290
            draw.rectangle((30,y-10,1620,y+270),outline='#cccccc',width=2)
            draw.text((45,y), f"Slot {slot['position']} | {slot['objectLayerSource']}",font=small,fill='black')
            for key,x in [('debugCropPath',45),('debugPokemonCropPath',410)]:
                crop = Image.open(slot[key]).convert('RGB')
                crop.thumbnail((330,210))
                page.paste(crop,(x,y+40))
            for rank,hit in enumerate(slot['pokemonTopCandidates']):
                draw.text((800,y+20+rank*37),f"{rank+1}. {hit['label']}  {hit['confidence']:.1%}",font=font,fill='black')
            draw.text((800,y+220),f"Decision: {slot['matchReason']}",font=small,fill='black')
        page.save(OUT / f'{path.stem}-review.jpg')
        pages.append(page)
        print([(s['position'],s['name'],s['confidence']) for s in result['detectedTeam']],flush=True)
    pages[0].save(OUT / 'prediction_crop_report.pdf',save_all=True,append_images=pages[1:],resolution=120)
    print('Report saved:', OUT, flush=True)

if __name__ == '__main__':
    main()
