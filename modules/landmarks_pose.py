import mediapipe as mp
import numpy as np
import cv2

from modules.utils import rotation_matrix_to_angles

FACE_3D_MODEL = np.array([
    [285, 528, 200],  # Nose
    [285, 371, 152],  # Forehead
    [197, 574, 128],  # Left lip corner
    [173, 425, 108],  # Left eye outer corner
    [360, 574, 128],  # Right lip corner
    [391, 425, 108]   # Right eye outer corner
], dtype=np.float64)
# Corresponding landmark indices in MediaPipe
FACE_LANDMARK_IDS = [1, 9, 57, 130, 287, 359]

class LandmarkPoseEstimator:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

    def process_frame(self, frame):
        res = self.face_mesh.process(frame)
        if not res.multi_face_landmarks:
            return None, None
        lm = res.multi_face_landmarks[0]
        landmarks = [(p.x, p.y, p.z) for p in lm.landmark]
        yaw, pitch, roll = self._estimate_head_pose(lm.landmark, frame.shape)
        return landmarks, {"yaw": yaw, "pitch": pitch, "roll": roll}

    def _estimate_head_pose(self, landmarks, frame_shape):
        # Адаптировано из https://github.com/shenasa-ai/head-pose-estimation/blob/main/estimator.py
        h, w, _ = frame_shape
        face_2d = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in FACE_LANDMARK_IDS],
            dtype=np.float64
        )

        # Camera matrix
        focal_length = w
        cam_matrix = np.array([[focal_length, 0, w / 2],
                               [0, focal_length, h / 2],
                               [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(
            FACE_3D_MODEL,
            face_2d,
            cam_matrix,
            dist_matrix
        )
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        pitch, yaw, roll = rotation_matrix_to_angles(rot_mat)

        return yaw, pitch, roll
