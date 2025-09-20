import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import numpy as np
import cv2 # OpenCV 추가

class FeatureExtractor:
    """
    Hugging Face Vision Transformer (ViT) 모델을 사용하여
    이미지에서 전역 특징 벡터(임베딩)를 추출하는 클래스. (후보군 탐색용)
    """
    def __init__(self, model_name='google/vit-base-patch16-224'):
        """
        모델과 이미지 프로세서를 초기화하고 로드합니다.
        """
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"✅ FeatureExtractor initialized on device: {self.device}")

    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """
        주어진 PIL Image 객체로부터 768차원의 전역 임베딩 벡터를 추출합니다.
        """
        image = image.convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        avg_embedding = last_hidden_state.mean(dim=1).squeeze()
        return avg_embedding.cpu().numpy()

class LocalFeatureExtractor:
    def __init__(self):
        # --- [ACCURACY-UP] nfeatures 제한을 제거하여 더 많은 특징점 사용 ---
        self.orb = cv2.ORB_create()

    def get_features(self, image: Image.Image):
        frame_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(frame_gray, None)
        return keypoints, descriptors