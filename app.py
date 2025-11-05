import cv2, json
import argparse

from modules.face_detection import FaceDetector
from modules.face_recognition_module import FaceIdentifier
from modules.landmarks_pose import LandmarkPoseEstimator
from modules.lipreading import LipReader


parser = argparse.ArgumentParser(description="Face analysis MVP")
parser.add_argument(
    "--input",
    type=str,
    default='data/input.mp4',
    help="Input source: path to video file (default: 'data/input.mp4')."
)
args = parser.parse_args()


def main(input):
    cap = cv2.VideoCapture(input)
    face_detector = FaceDetector()
    face_id = FaceIdentifier("gallery")
    pose_estimator = LandmarkPoseEstimator()
    lip_reader = LipReader()

    frames_data = []
    frames = []
    landmark_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_bbox = face_detector.detect(frame)
        identity = face_id.identify(frame, face_bbox)
        landmarks, head_pose = pose_estimator.process_frame(frame)
        landmark_list.append(landmarks)
        frames.append(frame)

        frames_data.append({
            "identity": identity,
            "landmarks": landmarks,
            "head_pose": head_pose
        })

    lip_text = lip_reader.predict(frames, landmark_list)

    output = {"frames": frames_data, "lip_text": lip_text}
    json.dump(output, open("data/output.json", "w"), indent=2)


if __name__ == "__main__":
    main(args.input)
