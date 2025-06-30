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

# Serve Laravel public assets
# laravel_public_path = os.getenv("LARAVEL_PUBLIC_PATH")
# app.mount("/laravel-static", StaticFiles(directory=laravel_public_path), name="laravel-static")

# app.add_middleware(CheckAuthenticationMiddleware)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/logs", response_class=HTMLResponse)
async def serve_logs():
    with open("static/logs.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@router.get("/logs/json")
async def get_logs_json():
    try:
        with open("static/logs/chatbot_logs.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"logs": []}
    
app.include_router(router)
