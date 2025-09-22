import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
import cv2
import numpy as np
from typing import Tuple, List


class FeatureExtractor:
    def __init__(self, model_path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(
            self.device
        )

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def get_embedding(self, img: Image.Image) -> List[float]:
        """Extracts a feature embedding from an image."""
        # FIX: Convert image to RGB to handle different channel formats (e.g., RGBA, Grayscale)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()


class LocalFeatureExtractor:
    def __init__(self):
        self.orb = cv2.ORB_create()

    def get_features(self, image: Image.Image) -> Tuple[Tuple[cv2.KeyPoint, ...], np.ndarray | None]:
        """Extracts local features (keypoints and descriptors) from an image."""
        # FIX: Convert image to RGB before converting to Grayscale
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        frame_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(frame_gray, None)
        return keypoints, descriptors