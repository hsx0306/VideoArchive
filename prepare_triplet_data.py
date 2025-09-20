import os
import random
import cv2
from PIL import Image
import argparse
from tqdm import tqdm
import shutil

def get_frame(video_path, timestamp_sec):
    """
    지정된 비디오 경로와 시간(초)을 기반으로 프레임을 추출하여 PIL 이미지로 반환합니다.
    (기존 video_processor.py의 함수를 독립적으로 재구성)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 비디오 파일을 열 수 없습니다 - {video_path}")
        return None

    # 타임스탬프(초)를 밀리초로 변환하여 프레임 위치 설정
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    
    success, frame = cap.read()
    cap.release()

    if success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    else:
        # print(f"오류: {os.path.basename(video_path)} 영상의 {timestamp_sec:.2f}초 지점에서 프레임을 가져오지 못했습니다.")
        return None

def get_video_duration(video_path):
    """OpenCV를 사용하여 비디오의 총 길이를 초 단위로 반환합니다."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration

def create_dataset(video_dir, output_dir, num_triplets_per_video, positive_delta):
    """
    영상들을 사용하여 Triplet 데이터셋을 생성하고 이미지 파일로 저장합니다.
    """
    print(f"📁 비디오 소스: {video_dir}")
    if not os.path.exists(video_dir):
        print(f"오류: '{video_dir}' 디렉토리를 찾을 수 없습니다.")
        return

    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if len(video_files) < 2:
        print("오류: Triplet 생성을 위해서는 최소 2개 이상의 비디오 파일이 필요합니다.")
        return

    # 기존 출력 디렉토리가 있으면 삭제하고 새로 생성
    if os.path.exists(output_dir):
        print(f"⚠️ 기존 '{output_dir}' 디렉토리를 삭제하고 새로 생성합니다.")
        shutil.rmtree(output_dir)
        
    anchor_dir = os.path.join(output_dir, "anchor")
    positive_dir = os.path.join(output_dir, "positive")
    negative_dir = os.path.join(output_dir, "negative")
    os.makedirs(anchor_dir)
    os.makedirs(positive_dir)
    os.makedirs(negative_dir)

    print(f"🚀 Triplet 데이터셋 생성을 시작합니다. -> {output_dir}")
    
    triplet_count = 0
    total_triplets = len(video_files) * num_triplets_per_video
    
    with tqdm(total=total_triplets, desc="Generating Triplets") as pbar:
        for anchor_video_file in video_files:
            other_videos = [v for v in video_files if v != anchor_video_file]

            anchor_video_path = os.path.join(video_dir, anchor_video_file)
            anchor_duration = get_video_duration(anchor_video_path)

            if anchor_duration < positive_delta * 2:
                pbar.update(num_triplets_per_video)
                continue

            for _ in range(num_triplets_per_video):
                # 1. Anchor & Positive 프레임 선택
                anchor_ts = random.uniform(0, anchor_duration - positive_delta)
                positive_ts = anchor_ts + random.uniform(0.1, positive_delta)

                # 2. Negative 프레임 선택
                negative_video_file = random.choice(other_videos)
                negative_video_path = os.path.join(video_dir, negative_video_file)
                negative_duration = get_video_duration(negative_video_path)
                if negative_duration == 0:
                    pbar.update(1)
                    continue
                negative_ts = random.uniform(0, negative_duration)
                
                # 3. 프레임 추출
                anchor_img = get_frame(anchor_video_path, anchor_ts)
                positive_img = get_frame(anchor_video_path, positive_ts)
                negative_img = get_frame(negative_video_path, negative_ts)
                
                # 4. 이미지 저장
                if anchor_img and positive_img and negative_img:
                    file_prefix = f"{triplet_count:06d}"
      
                    # Corrected code
                    video_name_no_ext = os.path.splitext(anchor_video_file)[0].replace('.', '_')
                    anchor_img_name = f"{file_prefix}_anchor_{video_name_no_ext}_{anchor_ts:.2f}.jpg"

                    video_name_no_ext_pos = os.path.splitext(anchor_video_file)[0].replace('.', '_')
                    positive_img_name = f"{file_prefix}_positive_{video_name_no_ext_pos}_{positive_ts:.2f}.jpg"

                    video_name_no_ext_neg = os.path.splitext(negative_video_file)[0].replace('.', '_')
                    negative_img_name = f"{file_prefix}_negative_{video_name_no_ext_neg}_{negative_ts:.2f}.jpg"

                    # I've also noticed a small bug where the positive_img_name was using the anchor_video_file variable, it is now corrected. I also fixed the way timestamps are formatted in filenames.
                    video_name_no_ext = os.path.splitext(anchor_video_file)[0].replace('.', '_')
                    anchor_ts_str = f"{anchor_ts:.2f}".replace('.', '_')
                    anchor_img_name = f"{file_prefix}_anchor_{video_name_no_ext}_{anchor_ts_str}.jpg"

                    positive_ts_str = f"{positive_ts:.2f}".replace('.', '_')
                    positive_img_name = f"{file_prefix}_positive_{video_name_no_ext}_{positive_ts_str}.jpg"

                    negative_video_name_no_ext = os.path.splitext(negative_video_file)[0].replace('.', '_')
                    negative_ts_str = f"{negative_ts:.2f}".replace('.', '_')
                    negative_img_name = f"{file_prefix}_negative_{negative_video_name_no_ext}_{negative_ts_str}.jpg"

                    anchor_img.save(os.path.join(anchor_dir, anchor_img_name))
                    positive_img.save(os.path.join(positive_dir, positive_img_name))
                    negative_img.save(os.path.join(negative_dir, negative_img_name))
                    
                    triplet_count += 1
                
                pbar.update(1)

    print(f"\n✅ Triplet 데이터셋 생성 완료! 총 {triplet_count}개의 유효한 샘플이 생성되었습니다.")

def main():
    parser = argparse.ArgumentParser(description="영상 파일로부터 Triplet 데이터셋을 생성하는 스크립트")
    parser.add_argument("--video_dir", type=str, default="backend/videos",
                        help="원본 영상 파일들이 있는 디렉토리 경로")
    parser.add_argument("--output_dir", type=str, default="triplet_dataset",
                        help="생성된 Triplet 데이터셋 이미지들을 저장할 디렉토리 경로")
    parser.add_argument("--num_triplets", type=int, default=100,
                        help="각 비디오 파일 당 생성할 Triplet 샘플의 수")
    parser.add_argument("--delta", type=float, default=2.0,
                        help="Anchor와 Positive 프레임 간의 최대 시간 간격 (초)")

    args = parser.parse_args()
    
    create_dataset(args.video_dir, args.output_dir, args.num_triplets, args.delta)

if __name__ == '__main__':
    main()