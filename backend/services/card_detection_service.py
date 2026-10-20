"""Validated full-photo card detection, NMS, and evidence-based layout recovery.

The existing slot YOLO detects cards; the existing inner-object YOLO and
MobileNet remain responsible for sprites and identity. Never use guide boxes.
"""
import logging
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from paths import CV_MODEL_DIR

MODEL_PATH = CV_MODEL_DIR / 'slot_detector/yolov8n_slots_mixed_synth_real_cv/weights/best.pt'
logger = logging.getLogger(__name__)
NORMAL_CONFIDENCE = 0.30
RETRY_CONFIDENCE = 0.12
NORMAL_SIZE = 960
RETRY_SIZE = 1536


class CardCrops(list):
    """Keep the existing list API while carrying zero-detection diagnostics."""
    def __init__(self, detection):
        super().__init__()
        self.detection = detection


@lru_cache(maxsize=1)
def load_card_model():
    if not MODEL_PATH.exists():
        logger.warning('Card YOLO unavailable: %s', MODEL_PATH)
        return None
    from ultralytics import YOLO
    return YOLO(str(MODEL_PATH))


def bounded_box(box, shape):
    h, w = shape[:2]
    x1 = max(0, min(w, int(round(box['x']))))
    y1 = max(0, min(h, int(round(box['y']))))
    x2 = max(0, min(w, int(round(box['x'] + box['width']))))
    y2 = max(0, min(h, int(round(box['y'] + box['height']))))
    if x2 <= x1 or y2 <= y1:
        return None
    return {**box, 'x': x1, 'y': y1, 'width': x2-x1, 'height': y2-y1}


def overlap(a, b):
    area = max(0, min(a['x']+a['width'], b['x']+b['width'])-max(a['x'], b['x'])) * max(
        0, min(a['y']+a['height'], b['y']+b['height'])-max(a['y'], b['y']))
    aa, ab = a['width']*a['height'], b['width']*b['height']
    return area / max(1, aa+ab-area), area / max(1, min(aa, ab))


def nms(boxes, threshold=0.45):
    kept = []
    # Original detections win over retries/recovery, even if recovery scores higher.
    for box in sorted(boxes, key=lambda b: (b.get('source') == 'yolo', b.get('confidence', 0)), reverse=True):
        if not any(overlap(box, old)[0] > threshold or overlap(box, old)[1] > 0.65 for old in kept):
            kept.append(box)
    return sorted(kept, key=lambda b: b['y']+b['height']/2)


def red_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (((hsv[:, :, 0] <= 14) | (hsv[:, :, 0] >= 145)) &
            (hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 45)).astype(np.uint8)


def evidence(box, mask):
    """Require a horizontal red card with substantial visible area."""
    h, w = mask.shape
    width, height = box['width'], box['height']
    if width < max(20, w*0.08) or height < max(12, h*0.018) or not 1.35 <= width/height <= 7:
        return 0.0
    crop = mask[box['y']:box['y']+height, box['x']:box['x']+width]
    fill = float(crop.mean()) if crop.size else 0
    # A card must have red support across most rows and columns, not one UI icon.
    row_support = float((crop.mean(axis=1) > 0.25).mean())
    column_support = float((crop.mean(axis=0) > 0.25).mean())
    return fill if fill >= 0.35 and row_support >= 0.72 and column_support >= 0.65 else 0.0


def yolo_candidates(image, confidence, image_size):
    model = load_card_model()
    if model is None:
        return []
    results = model.predict(image, conf=confidence, imgsz=image_size, iou=0.45, verbose=False)
    boxes = []
    for result in results:
        for item in result.boxes:
            if result.names.get(int(item.cls[0])) != 'opponent_slot':
                continue
            x1, y1, x2, y2 = item.xyxy[0].tolist()
            boxes.append(dict(x=x1, y=y1, width=x2-x1, height=y2-y1,
                              confidence=float(item.conf[0]), source='yolo' if confidence == NORMAL_CONFIDENCE else 'yolo_retry'))
    return boxes


def runs(active):
    padded = np.pad(np.asarray(active, dtype=np.int8), (1, 1))
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def image_candidates(mask, anchors):
    """Find card edges in stripes anchored to observed card columns.

    Stripe voting follows local card geometry rather than image coordinates.
    Narrow stripes avoid sprites, type icons, and red background connections.
    Without anchors, connected red components seed possible columns.
    """
    h, w = mask.shape
    seeds = list(anchors)
    if not seeds:
        count, _labels, stats, _centers = cv2.connectedComponentsWithStats(mask, 8)
        seeds = [dict(x=int(x), y=int(y), width=int(bw), height=int(bh))
                 for x,y,bw,bh,area in stats[1:] if area > w*h*0.003 and bw > w*0.12]
    candidates = []
    for seed in seeds:
        for fraction in (0.08, 0.20, 0.75):
            left = max(0, int(seed['x']+seed['width']*fraction))
            right = min(w, left+max(4, int(seed['width']*0.08)))
            if right <= left:
                continue
            profile = mask[:, left:right].mean(axis=1).astype(np.float32)
            profile = cv2.GaussianBlur(profile[:, None], (1, 5), 0).ravel()
            for top, bottom in runs(profile >= 0.65):
                height = int(bottom-top)
                if height < h*0.025 or height > h*0.24:
                    continue
                # Evidence must extend horizontally through the stripe.
                support = mask[top:bottom].mean(axis=0) >= 0.38
                for x1,x2 in runs(support):
                    if x1 <= left < x2 and x2-x1 >= w*0.12:
                        box = dict(x=int(x1), y=int(top), width=int(x2-x1), height=height,
                                   source='image_evidence', confidence=0.0)
                        score = evidence(box, mask)
                        if score:
                            box['confidence'] = round(score, 4)
                            candidates.append(box)
    return nms(candidates)


def aligned(box, anchors):
    if not anchors:
        return True
    # Pairwise local alignment supports changing size and moderate perspective.
    nearest = min(anchors, key=lambda a: abs(a['y']-box['y']))
    ratio = box['width']/nearest['width']
    drift = abs(box['x']+box['width']/2-nearest['x']-nearest['width']/2)/nearest['width']
    return 0.60 <= ratio <= 1.65 and drift <= 0.35 and 0.55 <= box['height']/nearest['height'] <= 1.7


def recover_layout(boxes, mask):
    """Accept inferred positions only when a separate image-backed box exists."""
    anchors = list(boxes)
    candidates = image_candidates(mask, anchors)
    if not anchors:
        # Select a coherent column; never pad it to six rows.
        groups = [[b for b in candidates if aligned(b, [seed])] for seed in candidates]
        return max(groups, key=lambda g: (len(g), sum(b['width']*b['height'] for b in g)), default=[])
    recovered = list(anchors)
    # Revisit evidence candidates after extending either end of the column.
    # This also works when only the last (rather than the first) card is found.
    for _pass in range(6):
        previous_count = len(recovered)
        for candidate in candidates:
            if len(recovered) >= 6:
                break
            if aligned(candidate, recovered) and not any(overlap(candidate, old)[1] > 0.10 for old in recovered):
                # Observed dimensions/alignment + a bounded gap in the observed layout.
                centers = sorted(b['y']+b['height']/2 for b in recovered)
                cy = candidate['y']+candidate['height']/2
                typical_height = float(np.median([b['height'] for b in recovered]))
                distance = min(abs(cy-y) for y in centers)
                if 0.65*typical_height <= distance <= 2.5*typical_height:
                    recovered.append({**candidate, 'source': 'layout_recovered'})
        if len(recovered) == previous_count:
            break
    return nms(recovered)


def assign_positions(boxes):
    """Preserve internal gaps; end-slot placement is explicitly ambiguous."""
    boxes = sorted(boxes, key=lambda b: b['y']+b['height']/2)
    if not boxes:
        return [], list(range(1,7)), True
    heights = [b['height'] for b in boxes]
    gaps = np.diff([b['y']+b['height']/2 for b in boxes])
    base = float(np.median(heights))*1.10
    close_gaps = [g for g in gaps if 0.7*base <= g <= 1.5*base]
    pitch = float(np.median(close_gaps)) if close_gaps else base
    position = 1
    assigned = []
    for index, box in enumerate(boxes):
        if index:
            position += max(1, int(round(gaps[index-1]/pitch)))
        if position > 6:
            break
        assigned.append({**box, 'position': position, 'validated': True})
    unresolved = sorted(set(range(1,7))-{b['position'] for b in assigned})
    # With fewer than six, there is no evidence distinguishing missing leading
    # from trailing cards. Positions are relative to the first visible card.
    return assigned, unresolved, bool(unresolved)


def detect_cards(image):
    if image is None or not image.size:
        return dict(boxes=[], unresolvedPositions=list(range(1,7)), positionsAreRelative=True, retried=False)
    mask = red_mask(image)
    def validate(candidates):
        valid = []
        for candidate in candidates:
            box = bounded_box(candidate, image.shape)
            if box is not None and evidence(box, mask):
                valid.append(box)
        return nms(valid)
    first = validate(yolo_candidates(image, NORMAL_CONFIDENCE, NORMAL_SIZE))
    retried = len(first) < 6
    boxes = first
    if retried:
        boxes = validate(first + yolo_candidates(image, RETRY_CONFIDENCE, RETRY_SIZE))
    if len(boxes) < 6:
        boxes = recover_layout(boxes, mask)
    # Reject significant overlaps rather than feeding ambiguous cards to identity.
    clean = []
    for box in sorted(boxes, key=lambda b: (b.get('source') == 'yolo', b['confidence']), reverse=True):
        if not any(overlap(box, other)[1] > 0.10 for other in clean):
            clean.append(box)
    ordered, unresolved, relative = assign_positions(clean)
    return dict(boxes=ordered, unresolvedPositions=unresolved, positionsAreRelative=relative,
                retried=retried, initialCount=len(first), validatedCount=len(ordered))


def save_overlay(image, result, path):
    overlay = image.copy()
    for box in result['boxes']:
        recovered = box['source'] not in {'yolo', 'yolo_retry'}
        color = (0,180,255) if recovered else (70,220,70)
        x,y,w,h = (box[k] for k in ('x','y','width','height'))
        cv2.rectangle(overlay,(x,y),(x+w,y+h),color,max(2,image.shape[1]//500))
        caption = f"{box['position']} {box['source']} {box['confidence']:.2f}"
        scale = max(.5,image.shape[1]/2200)
        (tw,th),baseline = cv2.getTextSize(caption,cv2.FONT_HERSHEY_SIMPLEX,scale,2)
        tx = max(0,min(x,image.shape[1]-tw-8))
        ty = min(image.shape[0]-baseline-5,max(th+8,y+th+8))
        cv2.rectangle(overlay,(tx,ty-th-5),(tx+tw+6,ty+baseline+3),(20,20,20),-1)
        cv2.putText(overlay,caption,(tx+3,ty),cv2.FONT_HERSHEY_SIMPLEX,
                    scale,color,2,cv2.LINE_AA)
    cv2.putText(overlay,'Unresolved: '+str(result['unresolvedPositions']),(20, image.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX,max(.5,image.shape[1]/2200),(0,180,255),2,cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path),overlay):
        raise OSError(f'Could not save card overlay: {path}')
