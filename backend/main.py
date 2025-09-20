from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles # StaticFiles 임포트
import uvicorn
from PIL import Image
import io
import traceback
import os

from .search_engine import SearchEngine
from .video_processor import build_index

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 비디오 파일이 저장된 디렉토리를 /videos 경로로 서빙
# video_processor.py에서 사용하는 VIDEO_DIR 경로를 실제 경로로 지정해야 합니다.
# 예: "./videos"
VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos") 

app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")


try:
    search_engine = SearchEngine()
except Exception as e:
    print(f"SearchEngine 초기화 실패: {e}")
    search_engine = None

@app.post("/search")
async def search_scene(file: UploadFile = File(...)):
    if search_engine is None:
        raise HTTPException(status_code=503, detail="서버가 아직 준비되지 않았습니다. 인덱스를 확인하세요.")
        
    try:
        contents = await file.read()
        query_image = Image.open(io.BytesIO(contents))
        
        # 검색 수행 (결과 객체 구조 변경됨)
        result = search_engine.search(query_image)

        if result:
            # 확장된 결과 객체를 그대로 반환
            return {"success": True, "result": result}
        else:
            return {"success": False, "message": "유사한 장면을 찾지 못했습니다."}
            
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"검색 중 오류가 발생했습니다: {e}")


@app.post("/index")
async def trigger_indexing():
    try:
        build_index()
        global search_engine
        search_engine = SearchEngine()
        return {"success": True, "message": "영상 인덱싱이 성공적으로 완료되었습니다."}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"인덱싱 중 오류가 발생했습니다: {e}")

@app.get("/")
def read_root():
    return {"message": "Video Scene Search API가 실행 중입니다."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)