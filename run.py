from Pipeline import Pipeline


pipeline_structure = Pipeline()
pipeline_structure.cut_into_frames()
# pipeline_structure.show_frames()
# pipeline_structure.simple_opencv_facedetection("/usr/local/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml")
pipeline_structure.ssd_facedetection()
# pipeline_structure.mtcnn_facedetection()
pipeline_structure.show_faces(30)
