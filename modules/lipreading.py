import numpy as np
import torch
import argparse

from auto_avsr.lightning import ModelModule
from auto_avsr.preparation.detectors.mediapipe.detector import LandmarksDetector
from auto_avsr.preparation.detectors.mediapipe.video_process import VideoProcess
from auto_avsr.datamodule.transforms import VideoTransform


CKPT_PATH = "models/vsr_trlrs3_base.pth"


def preprocess_landmarks(shape, frames_lms):
    h, w, _ = shape
    landmarks = []
    for lms in frames_lms:
        keypoints = [
            lms[468][:2],  # Left eye
            lms[473][:2],  # Right eye
            lms[1][:2],  # Nose tip
            (np.array(lms[13][:2]) + np.array(lms[14][:2])) / 2  # Mouth center
        ]
        keypoints = np.array([[x*w, y*h] for x, y in keypoints])
        landmarks.append(keypoints)
    return landmarks


class InferencePipeline(torch.nn.Module):
    def __init__(self, ckpt_path):
        super(InferencePipeline, self).__init__()

        self.landmarks_detector = LandmarksDetector()
        self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform(subset="test")

        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        parser = argparse.ArgumentParser()
        args, _ = parser.parse_known_args(args=[])
        setattr(args, 'modality', 'video')
        self.modelmodule = ModelModule(args)
        self.modelmodule.model.load_state_dict(ckpt)
        self.modelmodule.eval()

    def forward(self, frames, landmarks):
        video = np.stack(frames, axis=0)
        keypoints = preprocess_landmarks(frames[0].shape, landmarks)
        video = self.video_process(video, keypoints)
        video = torch.tensor(video)
        video = video.permute((0, 3, 1, 2))
        video = self.video_transform(video)
        with torch.no_grad():
            transcript = self.modelmodule(video)

        return transcript


class LipReader:
    def __init__(self):
        self.pipeline = InferencePipeline(CKPT_PATH)


    def predict(self, frames, landmarks):
        transcript = self.pipeline(frames, landmarks)
        return transcript
