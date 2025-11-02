import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# 3D model reference points (in mm, roughly scaled)

face_3d_model = np.array([
    [285, 528, 200],  # Nose
    [285, 371, 152],  # Forehead
    [197, 574, 128],  # Left lip corner
    [173, 425, 108],  # Left eye outer corner
    [360, 574, 128],  # Right lip corner
    [391, 425, 108]   # Right eye outer corner
], dtype=np.float64)
# Corresponding landmark indices in MediaPipe
LANDMARK_IDS = [1, 9, 57, 130, 287, 359]

def rotation_matrix_to_angles(rotation_matrix):
    """
    Calculate Euler angles from rotation matrix.
    :param rotation_matrix: A 3*3 matrix with the following structure
    [Cosz*Cosy  Cosz*Siny*Sinx - Sinz*Cosx  Cosz*Siny*Cosx + Sinz*Sinx]
    [Sinz*Cosy  Sinz*Siny*Sinx + Sinz*Cosx  Sinz*Siny*Cosx - Cosz*Sinx]
    [  -Siny             CosySinx                   Cosy*Cosx         ]
    :return: Angles in degrees for each axis
    """
    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                     rotation_matrix[1, 0] ** 2))
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([x, y, z]) * 180. / math.pi

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        face_2d = np.array([(lm[i].x * w, lm[i].y * h) for i in LANDMARK_IDS], dtype=np.float64)

        # Camera matrix
        focal_length = w
        cam_matrix = np.array([[focal_length, 0, w / 2],
                               [0, focal_length, h / 2],
                               [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # SolvePnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d_model, face_2d, cam_matrix, dist_matrix)
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        pitch, yaw, roll = rotation_matrix_to_angles(rot_mat)

        # Display angles
        cv2.putText(frame, f"Pitch: {int(pitch)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Yaw:   {int(yaw)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Roll:  {int(roll)}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Draw line showing head direction
        nose_2d = face_2d[0].astype(int)
        nose_3d, _ = cv2.projectPoints(np.array([(0, 0, 10000.0)]), rot_vec, trans_vec, cam_matrix, dist_matrix)
        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_3d[0][0][0]), int(nose_3d[0][0][1]))
        cv2.line(frame, p1, p2, (255, 0, 255), 2)

    cv2.imshow("Head Pose Estimation", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
