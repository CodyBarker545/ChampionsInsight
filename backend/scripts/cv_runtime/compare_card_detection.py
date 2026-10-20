"""Replay a saved photo manifest and compare card/crop results without retraining.

Usage from repository root:
  .venv/Scripts/python.exe backend/scripts/cv_runtime/compare_card_detection.py \
    --baseline tmp/pdfs/mobile-originals/results.json \
    --output backend/data/cv/reports/card_detection_layout_20261019

Baseline entries contain source, file, expected (six visually annotated labels),
and result (the prior detection response). The original inputs are read-only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'backend'))
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw
from werkzeug.datastructures import FileStorage
from services import cv_service, opponent_detection_service as detection
from services.image_service import prepare_opponent_image_for_prediction


def safe(value):
    if isinstance(value,dict): return {k:safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [safe(v) for v in value]
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    return value


def scores(records):
    totals=dict(photos=len(records),sixCardPhotos=0,validCards=0,correct=0,wrong=0,review=0,unresolved=0)
    for record in records:
        result=record['result']
        count=result['quality']['metrics']['detectedCardCount']
        totals['sixCardPhotos']+=count==6
        totals['validCards']+=count
        totals['unresolved']+=len(result.get('unresolvedPositions',[]))
        for slot in result['detectedTeam']:
            position=slot['position']
            truth=record['expected'][position-1]
            key='review' if slot['needsReview'] else ('correct' if slot['name']==truth else 'wrong')
            totals[key]+=1
    totals['sixCardSuccessRate']=totals['sixCardPhotos']/max(1,len(records))
    return totals


def preview(record,path):
    page=Image.new('RGB',(1500,1900),'white')
    draw=ImageDraw.Draw(page)
    draw.text((25,15),record['file']+' | card crops / final classifier inputs / result',fill='black',font_size=24)
    for row,slot in enumerate(record['result']['detectedTeam']):
        y=65+row*290
        for key,x,bounds in [('debugCropPath',25,(700,240)),('debugPokemonCropPath',750,(230,240))]:
            crop=Image.open(slot[key]).convert('RGB');crop.thumbnail(bounds);page.paste(crop,(x,y+30))
        truth=record['expected'][slot['position']-1]
        text=f"Slot {slot['position']}\nExpected: {truth}\nApp: {slot['name']}\nScore: {slot['confidence']:.2%}\nCard: {slot['cardSource']}"
        draw.multiline_text((1020,y+35),text,fill='black',spacing=10,font_size=18)
    page.save(path,quality=90)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    baseline=json.loads(args.baseline.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'baseline.json').write_text(json.dumps(baseline,indent=2),encoding='utf-8')
    cv_service.OPPONENT_DEBUG_CROP_DIR=args.output/'crops'
    detection.OPPONENT_DEBUG_CROP_DIR=args.output/'crops'
    torch.set_num_threads(2)
    after=[]
    for old in baseline:
        source=Path(old['source'])
        with source.open('rb') as stream:
            upload=prepare_opponent_image_for_prediction(FileStorage(stream=stream,filename=source.name,content_type='image/jpeg'))
        started=time.perf_counter()
        result=detection.detect_opponent_team_mobilenetv3_from_image(upload['image'],image_name=source.name,save_debug=True)
        record={**old,'seconds':round(time.perf_counter()-started,2),'result':safe(result)}
        after.append(record)
        preview(record,args.output/(source.stem+'-review.jpg'))
        (args.output/'after.json').write_text(json.dumps(after,indent=2),encoding='utf-8')
        print(source.name,[(s['position'],s['name'],s['confidence']) for s in result['detectedTeam']],flush=True)
    summary={'before':scores(baseline),'after':scores(after)}
    (args.output/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# Card detection comparison','','Same original photos, upload decoder, MobileNet checkpoint, and visual labels.','',
           '| Metric | Before | After |','| --- | ---: | ---: |']
    for key in summary['before']:
        lines.append(f'| {key} | {summary["before"][key]} | {summary["after"][key]} |')
    lines+=['','Before card counts are the original red-detector counts; its old fallback still fabricated six crops.',
            'After counts contain only validated YOLO/recovered cards. Missing end-slot placement is explicitly relative.',
            'Overlay images: `crops/<photo>/card-detections.jpg`; individual card and sprite crops are in the same folders.',
            'Per-photo review images show all card and classifier crops. `after.json` includes confidence, source, and unresolved positions.',
            'This is a 15-photo development sample, not a held-out accuracy benchmark.']
    (args.output/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
