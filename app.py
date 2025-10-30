import cv2, json
from modules.face_recognition_module import FaceIdentifier
from modules.landmarks_pose import LandmarkPoseEstimator
# from modules.lipreading_module import LipReader

def main():
    cap = cv2.VideoCapture("data/input.mp4")
    face_id = FaceIdentifier("gallery")
    pose_estimator = LandmarkPoseEstimator()
    # lip_reader = LipReader()

    frames_data = []
    lip_frames = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        identities = face_id.identify(frame)
        landmarks, head_pose = pose_estimator.process_frame(frame)
        # lip_frames.append(extract_lip_roi(frame, landmarks))

        frames_data.append({
            "identity": identities[0][0] if identities else "unknown",
            "landmarks": landmarks,
            "head_pose": head_pose
        })

    # lip_text = lip_reader.predict(lip_frames)

    # output = {"frames": frames_data, "lip_text": lip_text}
    # json.dump(output, open("data/output.json", "w"), indent=2)

def extract_lip_roi(frame, landmarks):
    if not landmarks: return None
    # простая функция вырезания области губ по лендмаркам
    return frame

if __name__ == "__main__":
    main()
