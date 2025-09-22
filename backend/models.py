import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel


class FeatureExtractor:
    def __init__(self, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(
            self.device
        )

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def get_feature(self, img: Image.Image) -> list[float]:
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()


class LocalFeatureExtractor:
    def __init__(self):
        self.orb = cv2.ORB_create()

    def get_features(self, image: Image.Image):
        frame_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(frame_gray, None)
        return keypoints, descriptors