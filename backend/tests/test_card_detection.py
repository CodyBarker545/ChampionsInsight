"""Geometry, evidence, partial-result, and production crop contract checks."""
import cv2
import numpy as np
import pytest
from services import card_detection_service as cards
from services import cv_service
from services import opponent_detection_service as opponent


def scene(visible=range(6), shift=0, perspective=False):
    image = np.zeros((1100, 850, 3), dtype=np.uint8)
    boxes = []
    for i in visible:
        x = 130 + shift + (i*6 if perspective else 0)
        y = 30 + i*165
        width = 380 + (i*8 if perspective else 0)
        height = 130 + (i*2 if perspective else 0)
        if perspective:
            points = np.array([[x,y+8],[x+width,y],[x+width-8,y+height],[x+4,y+height]])
            cv2.fillConvexPoly(image, points, (15, 10, 210))
        else:
            cv2.rectangle(image,(x,y),(x+width-1,y+height-1),(15,10,210),-1)
        boxes.append(dict(x=x,y=y,width=width,height=height,confidence=.95,source='yolo'))
    return image, boxes


@pytest.mark.parametrize('count', range(7))
def test_zero_through_six_visible_cards_are_never_fabricated(monkeypatch, count):
    image, boxes = scene(range(count))
    calls=[]
    def predict(image, confidence, image_size):
        calls.append((confidence,image_size))
        return list(reversed(boxes))
    monkeypatch.setattr(cards,'yolo_candidates',predict)
    result=cards.detect_cards(image)
    assert len(result['boxes'])==count
    assert result['unresolvedPositions']==list(range(count+1,7))
    assert result['retried'] is (count<6)
    if count<6:
        assert calls==[(.30,960),(.12,1536)]
    assert [b['y'] for b in result['boxes']]==sorted(b['y'] for b in boxes)
    assert all(b['validated'] for b in result['boxes'])


@pytest.mark.parametrize('count', range(7))
def test_missing_yolo_detections_can_be_recovered_only_from_visible_cards(monkeypatch,count):
    image, boxes=scene()
    monkeypatch.setattr(cards,'yolo_candidates',lambda *args: boxes[:count])
    result=cards.detect_cards(image)
    assert len(result['boxes'])==6
    assert result['unresolvedPositions']==[]
    # Every accepted initial box is preserved exactly.
    for initial in boxes[:count]:
        assert any(all(b[k]==initial[k] for k in ('x','y','width','height')) for b in result['boxes'])


def test_retry_keeps_initial_boxes_and_uses_new_confidence_resolution(monkeypatch):
    image, boxes=scene()
    def predict(_image,confidence,image_size):
        if confidence==cards.NORMAL_CONFIDENCE:
            assert image_size==960
            return boxes[:3]
        assert confidence==.12 and image_size==1536
        return [{**box,'source':'yolo_retry','confidence':.6} for box in boxes]
    monkeypatch.setattr(cards,'yolo_candidates',predict)
    result=cards.detect_cards(image)
    assert len(result['boxes'])==6
    assert [b['source'] for b in result['boxes']]==['yolo']*3+['yolo_retry']*3


@pytest.mark.parametrize('anchor', range(6))
def test_recovery_can_extend_both_directions_from_one_visible_anchor(monkeypatch,anchor):
    image,boxes=scene()
    monkeypatch.setattr(cards,'yolo_candidates',lambda *args: [boxes[anchor]])
    assert len(cards.detect_cards(image)['boxes'])==6


def test_iou_nms_removes_duplicate_and_contained_detections(monkeypatch):
    image,boxes=scene()
    duplicate={**boxes[2],'x':boxes[2]['x']+3,'confidence':.8}
    contained={**boxes[3],'width':boxes[3]['width']-30,'confidence':.7}
    monkeypatch.setattr(cards,'yolo_candidates',lambda *args: boxes+[duplicate,contained])
    result=cards.detect_cards(image)
    assert len(result['boxes'])==6
    assert cards.overlap(boxes[2],duplicate)[0]>.9


@pytest.mark.parametrize('shift,perspective', [(-110,False),(220,False),(100,True)])
def test_layout_recovery_supports_shift_scale_and_perspective(monkeypatch,shift,perspective):
    image,boxes=scene(shift=shift,perspective=perspective)
    monkeypatch.setattr(cards,'yolo_candidates',lambda *args: [boxes[i] for i in (0,2,5)])
    result=cards.detect_cards(image)
    assert len(result['boxes'])==6
    assert [b['position'] for b in result['boxes']]==list(range(1,7))
    assert all(cards.evidence(b,cards.red_mask(image)) for b in result['boxes'])
    for a,b in zip(result['boxes'],result['boxes'][1:]):
        assert cards.overlap(a,b)[1]<=.10


def test_internal_missing_card_stays_unresolved_with_relative_position_metadata(monkeypatch):
    image,boxes=scene((0,1,3,4,5))
    monkeypatch.setattr(cards,'yolo_candidates',lambda *args: boxes)
    result=cards.detect_cards(image)
    assert [b['position'] for b in result['boxes']]==[1,2,4,5,6]
    assert result['unresolvedPositions']==[3]
    assert result['positionsAreRelative'] is True


def test_false_box_without_image_evidence_is_rejected(monkeypatch):
    image,boxes=scene((0,2))
    fake={**boxes[0],'y':195}
    monkeypatch.setattr(cards,'yolo_candidates',lambda *args: boxes+[fake])
    result=cards.detect_cards(image)
    assert len(result['boxes'])==2
    assert not any(b['y']==195 for b in result['boxes'])


def test_crop_clamping_and_debug_artifacts(monkeypatch,upload_dir):
    image,boxes=scene(range(1))
    boxes[0]={**boxes[0],'x':-15,'width':530}
    monkeypatch.setattr(cards,'yolo_candidates',lambda *args: boxes)
    monkeypatch.setattr(cv_service,'OPPONENT_DEBUG_CROP_DIR',upload_dir)
    crops=cv_service.crop_opponent_team_slots_from_image(image,'test',save_debug=True)
    assert len(crops)==1
    assert crops[0]['box']['x']==0
    assert crops[0]['image'].size
    assert (upload_dir/'test/card-detections.jpg').exists()
    assert (upload_dir/'test/opponent-slot-1.jpg').exists()
    assert crops.detection['unresolvedPositions']==[2,3,4,5,6]


def test_no_cards_sends_no_inputs_to_mobilenet(monkeypatch):
    monkeypatch.setattr(cards,'yolo_candidates',lambda *args: [])
    calls=[]
    monkeypatch.setattr(opponent,'predict_pokemon_crops',lambda crops,**kwargs: calls.append(crops) or [])
    result=opponent.detect_opponent_team_mobilenetv3_from_image(np.zeros((400,400,3),dtype=np.uint8))
    assert result['detectedTeam']==[]
    assert result['unresolvedPositions']==list(range(1,7))
    assert calls==[[]]


def test_bounded_box_rejects_outside_or_degenerate_candidates():
    assert cards.bounded_box(dict(x=500,y=2,width=10,height=30),(100,100,3)) is None
    assert cards.bounded_box(dict(x=2,y=2,width=-1,height=30),(100,100,3)) is None


@pytest.mark.parametrize('scale', [0.5, 1.6])
def test_layout_recovery_at_different_image_scales(monkeypatch,scale):
    image,boxes=scene()
    image=cv2.resize(image,None,fx=scale,fy=scale)
    anchors=[{**b,**{k:int(b[k]*scale) for k in ('x','y','width','height')}} for b in boxes[:4]]
    monkeypatch.setattr(cards,'yolo_candidates',lambda *args: anchors)
    result=cards.detect_cards(image)
    assert len(result['boxes'])==6
    assert all(0<=b['x']<b['x']+b['width']<=image.shape[1] for b in result['boxes'])
    assert all(0<=b['y']<b['y']+b['height']<=image.shape[0] for b in result['boxes'])


def test_significantly_overlapping_card_is_not_sent_to_classifier(monkeypatch):
    image,boxes=scene((0,1))
    # A false box spanning two adjacent cards can have a plausible shape and red fill.
    merged={**boxes[0],'height':300,'confidence':.65}
    monkeypatch.setattr(cards,'yolo_candidates',lambda *args: boxes+[merged])
    result=cards.detect_cards(image)
    assert len(result['boxes'])==2
    assert cards.overlap(*result['boxes'])[1]==0


def test_partial_edge_card_is_not_invented_from_a_small_red_fragment(monkeypatch):
    image,boxes=scene(range(5))
    cv2.rectangle(image,(130,1087),(510,1099),(15,10,210),-1)
    monkeypatch.setattr(cards,'yolo_candidates',lambda *args: boxes)
    result=cards.detect_cards(image)
    assert len(result['boxes'])==5
    assert result['unresolvedPositions']==[6]
