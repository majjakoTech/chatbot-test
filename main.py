import os
import json
from fastapi import FastAPI
from dotenv import load_dotenv
from api.routes import router
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create logs directory if it doesn't exist
os.makedirs("static/logs", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Index file not found</h1>", status_code=404)

@app.get("/logs", response_class=HTMLResponse)
async def serve_logs():
    try:
        with open("static/logs.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Logs file not found</h1>", status_code=404)

@app.get("/multi-image-test", response_class=HTMLResponse)
async def serve_multi_image_test():
    try:
        with open("static/multi-image-test.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Multi-image test file not found</h1>", status_code=404)

@router.get("/logs/json")
async def get_logs_json():
    try:
        # Ensure logs directory exists
        os.makedirs("static/logs", exist_ok=True)
        
        with open("static/logs/chatbot_logs.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Create an empty logs file if it doesn't exist
        empty_logs = {"logs": []}
        with open("static/logs/chatbot_logs.json", "w", encoding="utf-8") as f:
            json.dump(empty_logs, f, indent=2, ensure_ascii=False)
        return empty_logs
    except Exception as e:
        return {"error": str(e), "logs": []}
    
app.include_router(router)
