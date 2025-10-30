import mediapipe as mp
import numpy as np

class LandmarkPoseEstimator:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

    def process_frame(self, frame):
        res = self.face_mesh.process(frame)
        if not res.multi_face_landmarks:
            return None, None
        lm = res.multi_face_landmarks[0]
        landmarks = [(p.x, p.y, p.z) for p in lm.landmark]
        yaw, pitch, roll = self._estimate_head_pose(landmarks, frame.shape)
        return landmarks, {"yaw": yaw, "pitch": pitch, "roll": roll}

    def _estimate_head_pose(self, landmarks, frame_shape):
        # Адаптировано из https://www.kaggle.com/code/khaledashrafm3wad/head-pose-estimation-using-mediapipe
        nose_tip = landmarks[1]
        chin = landmarks[152]
        left_eye = landmarks[33]
        right_eye = landmarks[263]

        roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        pitch = np.degrees(np.arctan2(chin[1] - nose_tip[1], chin[2] - nose_tip[2]))
        yaw = np.degrees(np.arctan2(nose_tip[0] - chin[0], nose_tip[2] - chin[2]))
        ...
