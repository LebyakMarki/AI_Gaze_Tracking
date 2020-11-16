import cv2
import numpy as np
from Models import Head, Frame
from mtcnn.mtcnn import MTCNN


class Pipeline:
    video_structure = []

    def __init__(self, filename="test_videos/one_person_all_directions.mp4"):
        self.filename = filename

    def cut_into_frames(self):
        # Use this function to get frames form input video
        video_capture = cv2.VideoCapture(self.filename)
        success, image = video_capture.read()
        while success:
            success, image = video_capture.read()
            if success:
                new_frame = Frame(image, [])
                self.video_structure.append(new_frame)

    def simple_opencv_facedetection(self, path_to_cascade):
        for index, frame in enumerate(self.video_structure):
            image = frame.image
            classifier = cv2.CascadeClassifier(path_to_cascade)
            boxes_with_faces = classifier.detectMultiScale(image)
            heads = []
            if len(boxes_with_faces) == 0:
                continue
            frame.one_face_detected = True
            for box in boxes_with_faces:
                x, y, width, height = box
                new_head = Head(x, y, width, height)
                heads.append(new_head)
            frame.heads = heads

    def mtcnn_facedetection(self):
        detector = MTCNN()
        for index, frame in enumerate(self.video_structure):
            image = frame.image
            boxes_with_faces = detector.detect_faces(image)
            heads = []
            if len(boxes_with_faces) == 0:
                continue
            frame.one_face_detected = True
            for box in boxes_with_faces:
                x, y, width, height = box['box']
                new_head = Head(x, y, width, height)
                heads.append(new_head)
            frame.heads = heads

    def ssd_facedetection(self):
        detector = cv2.dnn.readNetFromCaffe("SSDetector/deploy.prototxt.txt",
                                            caffeModel="SSDetector/res10_300x300_ssd_iter_140000.caffemodel")
        for index, frame in enumerate(self.video_structure):
            image = frame.image
            origin_h, origin_w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            detector.setInput(blob)
            detections = detector.forward()
            heads = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    frame.one_face_detected = True
                    bounding_box = detections[0, 0, i, 3:7] * np.array([origin_w, origin_h, origin_w, origin_h])
                    x_start, y_start, x_end, y_end = bounding_box.astype('int')
                    new_head = Head(x_start, y_start, x_end - x_start, y_end - y_start)
                    heads.append(new_head)
            frame.heads = heads

    def show_faces(self, wait_key=10):
        for index, frame in enumerate(self.video_structure):
            frame_name = "Frame " + str(index)
            if frame.one_face_detected:
                for index2, head in enumerate(frame.heads):
                    head_name = ", Head " + str(index2)
                    cv2.imshow(frame_name + head_name, frame.image[head.y:head.y + head.h, head.x:head.x + head.w])
                    cv2.waitKey(wait_key)

    def show_frames(self, wait_key=10):
        # Created for debugging and testing
        for index, frame in enumerate(self.video_structure):
            frame_name = "Frame " + str(index)
            cv2.imshow(frame_name, frame.image)
            cv2.waitKey(wait_key)
