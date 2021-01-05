from Pipeline import Pipeline
from visualization import visualize

if __name__ == "__main__":
    print("> Lets go!")
    pipeline_structure = Pipeline("test_videos/two_people_two_directions.mp4")
    pipeline_structure.cut_into_frames()
    pipeline_structure.ssd_facedetection()
    pipeline_structure.get_directions()
    pipeline_structure.write_to_yaml()
    # pipeline_structure.show_frames()
    # pipeline_structure.simple_opencv_facedetection(
    #     "/usr/local/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml")
    # pipeline_structure.mtcnn_facedetection()
    # pipeline_structure.show_faces(30)
    # pipeline_structure.show_gaze_detection(250)
    print("> Finished analyzing!")
    visualize()
else:
    print("Sorry :(")
