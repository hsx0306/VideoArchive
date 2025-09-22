import os
import random
import shutil

import cv2
from tqdm import tqdm


def create_dataset(videos_dir="backend/videos", dataset_dir="triplet_dataset"):
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)

    video_files = [f for f in os.listdir(videos_dir) if f.endswith(".mp4")]
    if len(video_files) < 2:
        print("적어도 2개 이상의 영상이 필요합니다.")
        return

    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(videos_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 2:
            continue

        # Anchor, Positive 프레임 추출
        anchor_frame_idx = random.randint(0, frame_count - 2)
        positive_frame_idx = anchor_frame_idx + 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, anchor_frame_idx)
        ret, anchor_frame = cap.read()
        if not ret:
            continue

        ret, positive_frame = cap.read()
        if not ret:
            continue

        # Negative 프레임 추출
        other_videos = [v for v in video_files if v != video_file]
        negative_video_file = random.choice(other_videos)
        negative_video_path = os.path.join(videos_dir, negative_video_file)
        neg_cap = cv2.VideoCapture(negative_video_path)
        neg_frame_count = int(neg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if neg_frame_count < 1:
            neg_cap.release()
            continue
        negative_frame_idx = random.randint(0, neg_frame_count - 1)
        neg_cap.set(cv2.CAP_PROP_POS_FRAMES, negative_frame_idx)
        ret, negative_frame = neg_cap.read()
        neg_cap.release()
        if not ret:
            continue

        cap.release()

        # 이미지 저장
        base_filename = os.path.splitext(video_file)[0]
        anchor_img_path = os.path.join(
            dataset_dir, f"{base_filename}_anchor_{anchor_frame_idx}.png"
        )
        positive_img_path = os.path.join(
            dataset_dir, f"{base_filename}_positive_{positive_frame_idx}.png"
        )
        negative_img_path = os.path.join(
            dataset_dir, f"{base_filename}_negative_{negative_frame_idx}.png"
        )

        cv2.imwrite(anchor_img_path, anchor_frame)
        cv2.imwrite(positive_img_path, positive_frame)
        cv2.imwrite(negative_img_path, negative_frame)


if __name__ == "__main__":
    create_dataset()