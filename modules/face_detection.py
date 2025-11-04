import face_recognition


class FaceDetector:
    def detect(self, frame):
        faces = face_recognition.face_locations(frame)
        biggest_face = max(faces, key=lambda bb: (bb[2] - bb[0]) * (bb[3] - bb[1]))
        return biggest_face