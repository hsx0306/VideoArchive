from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from PIL import Image
import io
import traceback
import os

from .search_engine import SearchEngine
from .video_processor import build_index

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos")
app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")

search_engine: SearchEngine | None = None

@app.on_event("startup")
async def startup_event():
    """Initializes the SearchEngine when the application starts."""
    global search_engine
    try:
        search_engine = SearchEngine()
        print("SearchEngine initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize SearchEngine: {e}")
        search_engine = None

@app.on_event("shutdown")
async def shutdown_event():
    """Placeholder for any cleanup on shutdown."""
    print("Application shutting down.")

@app.post("/search")
async def search_scene(file: UploadFile = File(...)):
    """Handles the image search request."""
    if search_engine is None:
        raise HTTPException(status_code=503, detail="The server is not yet ready. Please check the index.")

    try:
        contents = await file.read()
        query_image = Image.open(io.BytesIO(contents))

        result = search_engine.search(query_image)

        if result:
            return {"success": True, "result": result}
        else:
            return {"success": False, "message": "No similar scenes were found."}

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during the search: {e}")

@app.post("/index")
async def trigger_indexing(background_tasks: BackgroundTasks):
    """Triggers the video indexing process in the background."""
    background_tasks.add_task(build_index_and_reload)
    return {"success": True, "message": "Video indexing has started in the background."}

def build_index_and_reload():
    """Function to run indexing and reload the search engine."""
    global search_engine
    try:
        build_index()
        # Re-initialize the search engine to load the new index
        search_engine = SearchEngine()
        print("Search engine reloaded with the new index.")
    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred during indexing: {e}")


@app.get("/")
def read_root():
    return {"message": "Video Scene Search API is running."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)