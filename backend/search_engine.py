import json
import cv2  # <--- 이 줄이 오류를 해결합니다.
import faiss
import numpy as np
from PIL import Image
import os

from .models import FeatureExtractor, LocalFeatureExtractor

class SearchEngine:
    def __init__(self, faiss_index_path='index.faiss', index_mapping_path='index_mapping.json'):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.faiss_index_path = os.path.join(base_dir, faiss_index_path)
        self.index_mapping_path = os.path.join(base_dir, index_mapping_path)

        if os.path.exists(self.faiss_index_path) and os.path.exists(self.index_mapping_path):
            self.faiss_index = faiss.read_index(self.faiss_index_path)
            with open(self.index_mapping_path, "r", encoding="utf-8") as f:
                self.index_mapping = json.load(f)
        else:
            self.faiss_index = None
            self.index_mapping = None
            print("Warning: Index or mapping file not found. Please run indexing.")

        self.feature_extractor = FeatureExtractor()
        self.local_feature_extractor = LocalFeatureExtractor()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def search(self, image: Image.Image, top_k=5):
        if self.faiss_index is None:
            raise RuntimeError("Faiss index is not loaded.")

        query_embedding = self.feature_extractor.get_embedding(image)
        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = self.faiss_index.search(query_embedding, top_k)

        candidate_info = [self.index_mapping[str(i)] for i in indices[0]]

        query_kps, query_des = self.local_feature_extractor.get_features(image)
        if query_des is None:
            return []

        scores = []
        for candidate in candidate_info:
            video_path = candidate["video_path"]
            timestamp = candidate["timestamp"]

            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            cap.release()

            if ret:
                candidate_image = Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                cand_kps, cand_des = self.local_feature_extractor.get_features(
                    candidate_image
                )

                if cand_des is not None and len(cand_des) > 0:
                    matches = self.bf.match(query_des, cand_des)
                    score = len(matches)
                    scores.append(score)
                else:
                    scores.append(0)
            else:
                scores.append(0)

        sorted_indices = np.argsort(scores)[::-1]
        final_results = [
            {
                "video_path": candidate_info[i]["video_path"],
                "timestamp": candidate_info[i]["timestamp"],
                "score": scores[i],
            }
            for i in sorted_indices
        ]

        return final_results