import cv2
import os
import sys
import time
import torch
import gaze_v1.utils
from gaze_v1.utils import normalize_face, draw_gaze
import argparse
import traceback
import numpy as np

from PIL import Image
from gaze_v1.models import gazenet
from gaze_v1.detector import FaceDetector


def get_side(x1, y1, x2, y2):
    # print([x1, y1])
    # print([x2, y2])
    if abs(x1-x2) < 45 and abs(y1-y2) < 45:
        return "center"
    if abs(x1-x2) < 45 and y2 > y1:
        return "down"
    if abs(x1-x2) < 45 and y2 < y1:
        return "up"
    if abs(y1-y2) < 45 and x2 < x1:
        return "right"
    if abs(y1-y2) < 45 and x2 > x1:
        return "left"
    if y2 > y1 and x2 < x1:
        return "down-right"
    if y2 > y1 and x2 > x1:
        return "down-left"
    if y2 < y1 and x2 < x1:
        return "up-right"
    if y2 < y1 and x2 > x1:
        return "up-left"
    return "none"


def get_gaze_point(image):
    face_detector = FaceDetector(device="cpu")
    model = gazenet.GazeNet("cpu")
    state_dict = torch.load("gaze_v1/models/weights/gazenet.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    orig_image = image
    frame = orig_image[:, :, ::-1]
    frame = cv2.flip(frame, 1)
    img_h, img_w, _ = np.shape(frame)
    display = frame.copy()
    faces, landmarks = face_detector.detect(Image.fromarray(frame))

    if len(faces) != 0:
        for f, lm in zip(faces, landmarks):
            # Confidence check
            if f[-1] > 0.98:
                # Crop and normalize face Face
                face, gaze_origin, m = normalize_face(lm, frame)
                # Predict gaze
                with torch.no_grad():
                    gaze = model.get_gaze(face)
                    gaze = gaze[0].data.cpu()
                    # Draw results
                display = cv2.circle(display, gaze_origin, 3, (0, 255, 0), -1)
                display, dx, dy = draw_gaze(display, gaze_origin, gaze, color=(255, 0, 0), thickness=2)
                direction = get_side(gaze_origin[0], gaze_origin[1], dx[0], dy[0])
                return direction, cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
                # print(direction)
                # cv2.imshow('Gaze Demo', cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
                # cv2.waitKey(0)


# get_gaze_point("photos/9.png")