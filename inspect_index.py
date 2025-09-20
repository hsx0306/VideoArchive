import faiss
import numpy as np
import os

# --- 설정 ---
# backend 폴더에 있는 index.faiss 파일의 경로
INDEX_FILE_PATH = os.path.join('backend', 'index.faiss')

def inspect_faiss_index(file_path):
    """
    FAISS 인덱스 파일을 로드하여 주요 정보를 출력하고,
    저장된 벡터 중 일부를 확인합니다.
    """
    if not os.path.exists(file_path):
        print(f"❌ 오류: '{file_path}' 파일을 찾을 수 없습니다.")
        print("먼저 '/index' API를 호출하여 인덱싱을 완료했는지 확인하세요.")
        return

    try:
        # 1. FAISS 인덱스 파일 로드
        print(f"🔍 '{file_path}' 파일을 로드합니다...")
        index = faiss.read_index(file_path)     
        print("✅ 파일 로드 성공!")
        print("-" * 30)

        # 2. 인덱스의 기본 정보 출력
        print("📊 인덱스 기본 정보:")
        # is_trained: 인덱스가 검색에 사용될 준비가 되었는지 여부
        print(f"  - 훈련 여부 (is_trained): {index.is_trained}")
        # ntotal: 인덱스에 저장된 총 벡터의 개수
        print(f"  - 총 벡터 수 (ntotal): {index.ntotal}")
        # d: 각 벡터의 차원
        print(f"  - 벡터 차원 (d): {index.d}")
        print("-" * 30)

        # 3. 인덱스에 저장된 실제 벡터 데이터 확인 (일부만)
        if index.ntotal > 0:
            print("🔢 저장된 벡터 데이터 확인 (처음 5개):")
            
            # reconstruct_n(start_id, num_vectors) 함수로 벡터를 복원
            # 너무 많은 벡터를 한 번에 불러오면 메모리 문제가 발생할 수 있으므로 주의
            num_to_show = min(5, index.ntotal)
            vectors = index.reconstruct_n(0, num_to_show)
            
            print(f"  - 복원된 벡터의 형태 (shape): {vectors.shape}")
            for i in range(num_to_show):
                # 벡터가 너무 길기 때문에 앞부분 일부만 출력
                print(f"  - 벡터 {i}: {vectors[i][:10]}...") 
        else:
            print("ℹ️ 인덱스에 저장된 벡터가 없습니다.")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    inspect_faiss_index(INDEX_FILE_PATH)