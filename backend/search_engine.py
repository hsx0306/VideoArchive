import json
import cv2
import faiss
import numpy as np
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor

from .models import FeatureExtractor, LocalFeatureExtractor

class SearchEngine:
    def __init__(self, faiss_index_path='index.faiss', index_mapping_path='index_mapping.json'):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.faiss_index_path = os.path.join(base_dir, faiss_index_path)
        self.index_mapping_path = os.path.join(base_dir, index_mapping_path)
        self.video_dir = os.path.join(base_dir, "videos")

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

    def _get_rerank_score(self, candidate, query_des):
        """Helper function to calculate the re-ranking score for a single candidate."""
        video_filename = candidate["id"]
        video_path = os.path.join(self.video_dir, video_filename)
        timestamp = float(candidate["timestamp"])

        if not os.path.exists(video_path):
            print(f"Warning: Video file not found at {video_path}")
            return -1  # Indicate that the file is missing

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()

        if ret:
            candidate_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            _, cand_des = self.local_feature_extractor.get_features(candidate_image)

            if cand_des is not None and len(cand_des) > 0:
                matches = self.bf.match(query_des, cand_des)
                return len(matches)
        return 0


    def search(self, image: Image.Image, top_k=5):
        if self.faiss_index is None:
            raise RuntimeError("Faiss index is not loaded.")

        query_embedding = self.feature_extractor.get_embedding(image)
        query_embedding = np.array([query_embedding], dtype="float32")
        _, indices = self.faiss_index.search(query_embedding, top_k)

        candidate_info = [self.index_mapping[i] for i in indices[0]]
        query_kps, query_des = self.local_feature_extractor.get_features(image)

        if query_des is None:
            return []

        scores = []
        with ThreadPoolExecutor() as executor:
            future_to_candidate = {executor.submit(self._get_rerank_score, c, query_des): c for c in candidate_info}
            scores = [future.result() for future in future_to_candidate]


        valid_results = [
            (candidate_info[i], scores[i])
            for i in range(len(scores))
            if scores[i] != -1
        ]

        sorted_results = sorted(valid_results, key=lambda item: item[1], reverse=True)

        return [
            {
                "video_id": result[0]["id"],
                "timestamp": result[0]["timestamp"],
                "score": result[1],
            }
            for result in sorted_results
        ]