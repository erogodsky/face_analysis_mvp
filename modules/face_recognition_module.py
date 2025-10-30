import face_recognition
import numpy as np
from pathlib import Path

class FaceIdentifier:
    def __init__(self, gallery_path):
        self.encodings, self.names = self._load_gallery(gallery_path)

    def _load_gallery(self, path):
        encodings, names = [], []
        for img_path in Path(path).glob("*.jpg"):
            img = face_recognition.load_image_file(img_path)
            enc = face_recognition.face_encodings(img)
            if enc:
                encodings.append(enc[0])
                names.append(img_path.stem)
        return encodings, names

    def identify(self, frame):
        faces = face_recognition.face_locations(frame, model='cnn')
        encs = face_recognition.face_encodings(frame, faces)
        results = []
        for face_loc, enc in zip(faces, encs):
            distances = face_recognition.face_distance(self.encodings, enc)
            idx = np.argmin(distances)
            name = self.names[idx] if distances[idx] < 0.6 else "unknown"

            results.append((name, face_loc))
        return results
