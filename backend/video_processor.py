import cv2
from PIL import Image
import os
from tqdm import tqdm
import faiss
import numpy as np
import json

# --- PySceneDetect 라이브러리 추가 ---
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

# FeatureExtractor를 models.py에서 가져옵니다.
from .models import FeatureExtractor

# --- 설정 ---
VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.faiss")
MAPPING_PATH = os.path.join(os.path.dirname(__file__), "index_mapping.json")
RESIZE_DIM = (224, 224) 

# --- 시간 포맷 변환 함수 (변경 없음) ---
def _parse_timestamp(timestamp_str: str) -> float:
    if ':' in str(timestamp_str):
        parts = str(timestamp_str).split(':')
        seconds = 0
        for i, part in enumerate(reversed(parts)):
            seconds += float(part) * (60 ** i)
        return seconds
    else:
        return float(timestamp_str)

# --- get_frame_from_video 함수 (변경 없음) ---
def get_frame_from_video(video_file: str, timestamp: (str or float)) -> Image.Image:
    video_path = os.path.join(VIDEO_DIR, video_file)
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    timestamp_sec = _parse_timestamp(timestamp)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    success, frame = cap.read()
    cap.release()
    if success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    else:
        return None

# --- build_index 함수 수정 (피드백 추가 및 threshold 설명) ---
def build_index():
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)
        print("비디오 디렉토리를 생성했습니다. 'videos' 폴더에 영상을 넣어주세요.")
        return

    all_video_files = {f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))}
    
    indexed_videos = set()
    all_embeddings = []
    # index_mapping을 딕셔너리가 아닌 리스트로 사용
    index_mapping = []
    
    if os.path.exists(MAPPING_PATH) and os.path.exists(INDEX_PATH):
        print("📖 기존 인덱스 파일을 읽어옵니다...")
        with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
            # 리스트로 로드
            index_mapping = json.load(f)
        for item in index_mapping:
            indexed_videos.add(item['id'])
        index = faiss.read_index(INDEX_PATH)
        all_embeddings = [index.reconstruct(i) for i in range(index.ntotal)]
        print(f"총 {len(indexed_videos)}개의 영상이 이미 인덱싱되어 있습니다.")

    videos_to_process = sorted(list(all_video_files - indexed_videos))

    if not videos_to_process:
        print("✅ 처리할 새로운 영상이 없습니다. 모든 영상이 최신 상태입니다.")
        return

    print(f"🎥 총 {len(videos_to_process)}개의 새로운 영상을 발견했습니다. 장면 기반 인덱싱을 시작합니다...")
    
    feature_extractor = FeatureExtractor()
    new_embeddings = []
    
    # 새로운 데이터를 추가할 임시 리스트
    new_mapping_items = []

    for video_file in videos_to_process:
        video_path = os.path.join(VIDEO_DIR, video_file)
        
        try:
            video = open_video(video_path)
            scene_manager = SceneManager()
            
            scene_manager.add_detector(ContentDetector(threshold=27.0))
            
            scene_manager.detect_scenes(video, show_progress=False)
            scene_list = scene_manager.get_scene_list()
            print(f"🎬 '{video_file}'에서 {len(scene_list)}개의 장면을 감지했습니다.")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"경고: '{video_file}'을 열 수 없습니다. 건너뜁니다.")
                continue

            for scene_start, scene_end in tqdm(scene_list, desc=f"Processing Scenes: {video_file}"):
                middle_timestamp_sec = scene_start.get_seconds() + \
                                     (scene_end.get_seconds() - scene_start.get_seconds()) / 2
                
                cap.set(cv2.CAP_PROP_POS_MSEC, middle_timestamp_sec * 1000)
                ret, frame = cap.read()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb).resize(RESIZE_DIM)
                    embedding = feature_extractor.get_embedding(pil_image)
                    new_embeddings.append(embedding)
                    # 리스트에 딕셔너리 추가
                    new_mapping_items.append({
                        "id": video_file,
                        "timestamp": f"{middle_timestamp_sec:.2f}"
                    })
            cap.release()

        except Exception as e:
            print(f"오류: '{video_file}' 처리 중 예외 발생 - {e}")
            continue

    print("\n✅ 모든 비디오의 장면 처리가 완료되었습니다. 인덱스 생성을 시작합니다...")

    if not new_embeddings:
        print("새로운 영상에서 추출된 특징 벡터가 없습니다. 인덱싱을 종료합니다.")
        return
        
    all_embeddings.extend(new_embeddings)
    # 기존 매핑 데이터와 새로운 매핑 데이터를 합침
    index_mapping.extend(new_mapping_items)

    embeddings_np = np.array(all_embeddings).astype('float32')
    dimension = embeddings_np.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    
    faiss.write_index(index, INDEX_PATH)
    
    # 최종 매핑 리스트를 파일에 저장
    with open(MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(index_mapping, f, ensure_ascii=False, indent=4)
        
    print(f"🎉 인덱싱 최종 완료! {len(new_embeddings)}개의 새로운 대표 프레임이 처리되었습니다.")
    print(f"   - 총 {index.ntotal}개의 프레임이 인덱스에 저장되었습니다.")
    print(f"   - 인덱스 파일 저장: {INDEX_PATH}")
    print(f"   - 매핑 파일 저장: {MAPPING_PATH}")