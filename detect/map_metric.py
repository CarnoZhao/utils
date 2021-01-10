from collections import defaultdict
from .det_metric.lib.utils import BBFormat, BBType
from .det_metric.lib.Evaluator import Evaluator
from .det_metric.lib.BoundingBox import BoundingBox
from .det_metric.lib.BoundingBoxes import BoundingBoxes
import numpy as np

def mAP(ground, target, detect):
    bbs = BoundingBoxes()
    for i, (y, bb) in enumerate(zip(ground, target)):
        for j in range(len(y)):
            bbs.addBoundingBox(BoundingBox(i, y[j], *bb[j], bbType = BBType.GroundTruth, format = BBFormat.XYX2Y2))
    for i, bb in enumerate(detect):
        for j in range(len(bb)):
            bbs.addBoundingBox(BoundingBox(i, bb[j][-1], *bb[j][:4], bbType = BBType.Detected, classConfidence = bb[j][-2], format = BBFormat.XYX2Y2))
    ev = Evaluator()
    v = 0
    for th in (0.1, 0.3, 0.5):
        v += np.nanmean([ap["AP"] for ap in ev.GetPascalVOCMetrics(bbs, th)])
    return v / 3

def mAP_per_class(ground, target, detect):
    bbs = BoundingBoxes()
    for i, (y, bb) in enumerate(zip(ground, target)):
        for j in range(len(y)):
            bbs.addBoundingBox(BoundingBox(i, y[j], *bb[j], bbType = BBType.GroundTruth, format = BBFormat.XYX2Y2))
    for i, bb in enumerate(detect):
        for j in range(len(bb)):
            bbs.addBoundingBox(BoundingBox(i, bb[j][-1], *bb[j][:4], bbType = BBType.Detected, classConfidence = bb[j][-2], format = BBFormat.XYX2Y2))
    ev = Evaluator()
    v = defaultdict(list)
    for th in (0.1, 0.3, 0.5):
        for ap in ev.GetPascalVOCMetrics(bbs, th):
            v[int(ap["class"])].append(ap["AP"])
    v = dict((k, np.nanmean(v)) for k, v in v.items())
    return v