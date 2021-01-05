import torch
from torch.autograd import Variable
import math
from PIL import Image
import numpy as np
from gaze_estimator.box_utils import nms, _preprocess


def run_first_stage(image, net, scale, threshold, device):
    width, height = image.size
    sw, sh = math.ceil(width*scale), math.ceil(height*scale)
    img = image.resize((sw, sh), Image.BILINEAR)
    img = np.asarray(img, 'float32')
    img = Variable(torch.FloatTensor(_preprocess(img)))
    with torch.no_grad():
        output = net(img.to(device))
    probs = output[1].data.cpu().numpy()[0, 1, :, :]
    offsets = output[0].data.cpu().numpy()
    boxes = _generate_bboxes(probs, offsets, scale, threshold)
    if len(boxes) == 0:
        return None
    keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
    return boxes[keep]


def _generate_bboxes(probs, offsets, scale, threshold):
    stride = 2
    cell_size = 12
    inds = np.where(probs > threshold)
    if inds[0].size == 0:
        return np.array([])
    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]
    bounding_boxes = np.vstack([
        np.round((stride*inds[1] + 1.0)/scale),
        np.round((stride*inds[0] + 1.0)/scale),
        np.round((stride*inds[1] + 1.0 + cell_size)/scale),
        np.round((stride*inds[0] + 1.0 + cell_size)/scale),
        score, offsets
    ])
    return bounding_boxes.T
