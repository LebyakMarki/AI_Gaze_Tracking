import numpy as np
import torch
from torch.autograd import Variable
from gaze_estimator.get_nets import PNet, RNet, ONet
from gaze_estimator.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from gaze_estimator.first_stage import run_first_stage


class FaceDetector(object):
    def __init__(self, device):
        self.device = device
        self.pnet = PNet().to(self.device)
        self.rnet = RNet().to(self.device)
        self.onet = ONet().to(self.device)
        self.onet.eval()

    def detect(self, image, min_face_size=20.0,
               thresholds=[0.6, 0.7, 0.8],
               nms_thresholds=[0.7, 0.7, 0.7]):
        with torch.no_grad():
            width, height = image.size
            min_length = min(height, width)
            min_detection_size = 12
            factor = 0.707
            scales = []
            m = min_detection_size / min_face_size
            min_length *= m
            factor_count = 0
            while min_length > min_detection_size:
                scales.append(m * factor ** factor_count)
                min_length *= factor
                factor_count += 1
            bounding_boxes = []
            for s in scales:
                boxes = run_first_stage(image, self.pnet, scale=s, threshold=thresholds[0], device=self.device)
                bounding_boxes.append(boxes)
            bounding_boxes = [i for i in bounding_boxes if i is not None]
            if len(bounding_boxes) == 0: return [], []
            bounding_boxes = np.vstack(bounding_boxes)
            keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
            img_boxes = get_image_boxes(bounding_boxes, image, size=24)
            img_boxes = Variable(torch.FloatTensor(img_boxes))
            output = self.rnet(img_boxes.to(self.device))
            offsets = output[0].data.cpu().numpy()
            probs = output[1].data.cpu().numpy()
            keep = np.where(probs[:, 1] > thresholds[1])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]
            keep = nms(bounding_boxes, nms_thresholds[1])
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
            img_boxes = get_image_boxes(bounding_boxes, image, size=48)
            if len(img_boxes) == 0:
                return [], []
            img_boxes = Variable(torch.FloatTensor(img_boxes))
            output = self.onet(img_boxes.to(self.device))
            landmarks = output[0].data.cpu().numpy()
            offsets = output[1].data.cpu().numpy()
            probs = output[2].data.cpu().numpy()
            keep = np.where(probs[:, 1] > thresholds[2])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]
            landmarks = landmarks[keep]
            width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
            height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
            xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
            landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
            landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]
            bounding_boxes = calibrate_box(bounding_boxes, offsets)
            keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
            bounding_boxes = bounding_boxes[keep]
            landmarks = landmarks[keep]
        return bounding_boxes, landmarks
