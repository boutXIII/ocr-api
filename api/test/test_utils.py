from api.vision.predictor import iou

def test_iou_overlap():
    a = (0, 0, 10, 10)
    b = (5, 5, 15, 15)

    score = iou(a, b)

    assert 0 < score < 1

def test_iou_no_overlap():
    a = (0, 0, 10, 10)
    b = (20, 20, 30, 30)

    assert iou(a, b) == 0
