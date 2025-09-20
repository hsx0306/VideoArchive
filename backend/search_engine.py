import faiss
import json
import numpy as np
from PIL import Image
import os
import cv2
import base64
import io

from .models import FeatureExtractor, LocalFeatureExtractor
from .video_processor import get_frame_from_video

class SearchEngine:
    def __init__(self, index_path='index.faiss', mapping_path='index_mapping.json'):
        self.base_dir = os.path.dirname(__file__)
        self.index_path = os.path.join(self.base_dir, index_path)
        self.mapping_path = os.path.join(self.base_dir, mapping_path)

        self.global_feature_extractor = FeatureExtractor()
        self.local_feature_extractor = LocalFeatureExtractor()
        
        self.index = None
        self.mapping = None

        if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
            print("💾 기존 인덱스와 매핑 파일을 로드합니다.")
            self.index = faiss.read_index(self.index_path)
            with open(self.mapping_path, 'r', encoding='utf-8') as f:
                self.mapping = json.load(f)
            print("✅ 인덱스 로드 완료.")
        else:
            print("⚠️ 경고: 인덱스 파일 또는 매핑 파일을 찾을 수 없습니다. '/index' API를 먼저 호출해주세요.")

    # --- [ACCURACY-UP] 후보군(candidates) 기본값을 50으로 복원 ---
    def search(self, query_image: Image.Image, top_n: int = 5, candidates: int = 50, min_match_count: int = 10):
        if self.index is None or self.mapping is None:
            raise RuntimeError("인덱스가 생성되지 않았습니다. 먼저 /index 엔드포인트를 호출하여 인덱싱을 수행하세요.")

        # 1단계: FAISS로 후보군 탐색
        query_embedding = self.global_feature_extractor.get_embedding(query_image)
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, candidates)

        if not indices.size:
            return None

        # 2단계: 로컬 특징점 매칭으로 최종 프레임 결정
        query_kp, query_des = self.local_feature_extractor.get_features(query_image)
        if query_des is None:
            return None

        all_matches = []
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for candidate_id in indices[0]:
            candidate_info = self.mapping[candidate_id]
            video_file = candidate_info["id"]
            timestamp = candidate_info["timestamp"]

            candidate_frame_pil = get_frame_from_video(video_file, timestamp)
            if candidate_frame_pil is None: continue

            frame_kp, frame_des = self.local_feature_extractor.get_features(candidate_frame_pil)
            if frame_des is None: continue

            matches = bf.match(query_des, frame_des)
            
            if len(matches) > min_match_count:
                all_matches.append({
                    "score": len(matches),
                    "video_file": video_file,
                    "timestamp": timestamp,
                    "matched_frame_pil": candidate_frame_pil
                })

        if not all_matches:
            return None

        sorted_matches = sorted(all_matches, key=lambda x: x['score'], reverse=True)
        top_results = sorted_matches[:top_n]

        final_results = []
        for result in top_results:
            buffered = io.BytesIO()
            result["matched_frame_pil"].save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            final_results.append({
                "video_file": result["video_file"],
                "timestamp": result["timestamp"],
                "score": result["score"],
                "matched_frame": "data:image/jpeg;base64," + img_str,
            })

        return final_results