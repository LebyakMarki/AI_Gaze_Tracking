import cv2
import torch
import numpy as np
from PIL import Image
from gaze_estimator.utils import normalize_face, draw_gaze
from gaze_estimator.models import gazenet
from gaze_estimator.detector import FaceDetector


def get_side(x1, y1, x2, y2):
    max_center_distance = 40
    if abs(x1 - x2) <= max_center_distance and abs(y1 - y2) <= max_center_distance:
        return "center"
    elif -max_center_distance <= (x2 - x1) <= max_center_distance < abs(y1 - y2):
        if y1 <= y2:
            return "down"
        else:
            return "up"
    else:
        if x1 >= x2:
            return "right"
        else:
            return "left"


def get_gaze_point(image):
    face_detector = FaceDetector(device="cpu")
    model = gazenet.GazeNet("cpu")
    state_dict = torch.load("gaze_estimator/models/weights/gazenet.pth", map_location="cpu")
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
                origin_coordinates = [gaze_origin[0], gaze_origin[1]]
                destination_coordinates = [np.int(dx[0]), np.int(dy[0])]

                return direction, cv2.cvtColor(display, cv2.COLOR_RGB2BGR), origin_coordinates, destination_coordinates
    return None, None, None, None
