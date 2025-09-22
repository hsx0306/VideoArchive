import cv2
from PIL import Image
import os
from tqdm import tqdm
import faiss
import numpy as np
import json
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import FeatureExtractor

# --- Settings ---
VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.faiss")
MAPPING_PATH = os.path.join(os.path.dirname(__file__), "index_mapping.json")
RESIZE_DIM = (224, 224)
# Adjust the number of workers based on your CPU cores
MAX_WORKERS = os.cpu_count() or 1


def _parse_timestamp(timestamp_str: str) -> float:
    """Converts a timestamp string to seconds."""
    if ':' in str(timestamp_str):
        parts = str(timestamp_str).split(':')
        seconds = sum(float(part) * (60 ** i) for i, part in enumerate(reversed(parts)))
        return seconds
    return float(timestamp_str)

def get_frame_from_video(video_file: str, timestamp: str | float) -> Image.Image | None:
    """Extracts a frame from a video at a specific timestamp."""
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
    return None

def process_video(video_file: str, feature_extractor: FeatureExtractor) -> list[tuple[np.ndarray, dict]]:
    """Processes a single video file to extract scene-based features."""
    video_path = os.path.join(VIDEO_DIR, video_file)
    results = []
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0))
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open '{video_file}'. Skipping.")
            return []

        for scene_start, scene_end in scene_list:
            middle_timestamp_sec = scene_start.get_seconds() + (scene_end.get_seconds() - scene_start.get_seconds()) / 2
            cap.set(cv2.CAP_PROP_POS_MSEC, middle_timestamp_sec * 1000)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb).resize(RESIZE_DIM)
                embedding = feature_extractor.get_embedding(pil_image)
                mapping_info = {"id": video_file, "timestamp": f"{middle_timestamp_sec:.2f}"}
                results.append((embedding, mapping_info))
        cap.release()
    except Exception as e:
        print(f"Error processing '{video_file}': {e}")
    return results


def build_index():
    """Builds or updates the FAISS index for the videos."""
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)
        print("Video directory created. Please add videos to the 'videos' folder.")
        return

    all_video_files = {f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))}

    indexed_videos = set()
    all_embeddings = []
    index_mapping = []

    if os.path.exists(MAPPING_PATH) and os.path.exists(INDEX_PATH):
        print("📖 Reading existing index files...")
        with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
            index_mapping = json.load(f)
        for item in index_mapping:
            indexed_videos.add(item['id'])
        index = faiss.read_index(INDEX_PATH)
        all_embeddings = [index.reconstruct(i) for i in range(index.ntotal)]
        print(f"A total of {len(indexed_videos)} videos are already indexed.")

    videos_to_process = sorted(list(all_video_files - indexed_videos))

    if not videos_to_process:
        print("✅ No new videos to process. All videos are up-to-date.")
        return

    print(f"🎥 Found {len(videos_to_process)} new videos. Starting scene-based indexing...")

    feature_extractor = FeatureExtractor()
    new_embeddings = []
    new_mapping_items = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_video, vf, feature_extractor): vf for vf in videos_to_process}
        for future in tqdm(as_completed(futures), total=len(videos_to_process), desc="Processing Videos"):
            try:
                video_results = future.result()
                for embedding, mapping_info in video_results:
                    new_embeddings.append(embedding)
                    new_mapping_items.append(mapping_info)
            except Exception as e:
                video_file = futures[future]
                print(f"An error occurred while processing {video_file}: {e}")

    if not new_embeddings:
        print("No feature vectors were extracted from the new videos. Ending indexing.")
        return

    print("\n✅ Scene processing for all videos is complete. Starting index creation...")

    all_embeddings.extend(new_embeddings)
    index_mapping.extend(new_mapping_items)

    embeddings_np = np.array(all_embeddings, dtype='float32')
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    faiss.write_index(index, INDEX_PATH)

    with open(MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(index_mapping, f, ensure_ascii=False, indent=4)

    print(f"🎉 Indexing complete! {len(new_embeddings)} new representative frames have been processed.")
    print(f"   - Total frames in index: {index.ntotal}")
    print(f"   - Index file saved to: {INDEX_PATH}")
    print(f"   - Mapping file saved to: {MAPPING_PATH}")