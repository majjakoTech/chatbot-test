import os
import json
import base64
from typing import Optional, List
from dotenv import load_dotenv
from fastapi import Query, APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage

from services.chatbot import get_results, graph_builder

load_dotenv()

router = APIRouter()

@router.get('/search')
async def search(q: str = Query(..., description="Search Query"), city_id: Optional[int] = Query(None, description="Optional City ID")):
    try:
        response = get_results(q)
        return JSONResponse(content={"query": q, "response": response})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post('/analyze-multiple-reports')
async def analyze_multiple_reports(
    images: List[UploadFile] = File(...),
    question: str = Form("Please analyze these feline CKD medical reports and provide comprehensive insights.")
):
    try:
        if not images:
            return JSONResponse(status_code=400, content={"error": "No images provided"})
        
        # Process all images
        image_data_list = []
        image_filenames = []
        
        for image in images:
            # Read and encode the image
            image_data = await image.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            
            # Determine image format
            content_type = image.content_type or "image/jpeg"
            data_url = f"data:{content_type};base64,{image_base64}"
            
            image_data_list.append(data_url)
            image_filenames.append(image.filename)
        
        # Create multimodal message with multiple images
        content_blocks = [{"type": "text", "text": question}]
        
        for data_url in image_data_list:
            content_blocks.append({
                "type": "image_url", 
                "image_url": {"url": data_url}
            })
        
        # Create multimodal message
        message = HumanMessage(content=content_blocks)
        
        # Process with the graph
        result = graph_builder.invoke({
            "messages": [message],
        })
        
        response_text = result["messages"][-1].content
        
        # Log the interaction
        log_file = "static/logs/chatbot_logs.json"
        log_entry = {
            "query": f"Multiple reports analysis: {question}",
            "response": response_text,
            "image_count": len(image_filenames),
            "image_filenames": image_filenames
        }
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "logs" not in data:
                    data["logs"] = []
        else:
            data = {"logs": []}

        data["logs"].append(log_entry)

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return JSONResponse(content={
            "query": question,
            "response": response_text,
            "image_count": len(image_filenames),
            "image_filenames": image_filenames
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post('/chat-with-multiple-images')
async def chat_with_multiple_images(
    message: str = Form(...),
    images: Optional[List[UploadFile]] = File(None)
):
    try:
        if images:
            # Handle multiple images + text
            image_data_list = []
            image_filenames = []
            
            for image in images:
                image_data = await image.read()
                image_base64 = base64.b64encode(image_data).decode("utf-8")
                content_type = image.content_type or "image/jpeg"
                data_url = f"data:{content_type};base64,{image_base64}"
                
                image_data_list.append(data_url)
                image_filenames.append(image.filename)
            
            # Create multimodal message with multiple images
            content_blocks = [{"type": "text", "text": message}]
            
            for data_url in image_data_list:
                content_blocks.append({
                    "type": "image_url", 
                    "image_url": {"url": data_url}
                })
            
            multimodal_message = HumanMessage(content=content_blocks)
            
            result = graph_builder.invoke({
                "messages": [multimodal_message],
            })
        else:
            # Handle text-only
            result = graph_builder.invoke({
                "messages": [HumanMessage(content=message)],
            })
            image_filenames = []
        
        response_text = result["messages"][-1].content
        
        # Log the interaction
        log_file = "static/logs/chatbot_logs.json"
        log_entry = {
            "query": message,
            "response": response_text,
            "has_images": len(image_filenames) > 0,
            "image_count": len(image_filenames),
            "image_filenames": image_filenames
        }
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "logs" not in data:
                    data["logs"] = []
        else:
            data = {"logs": []}

        data["logs"].append(log_entry)

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return JSONResponse(content={
            "query": message,
            "response": response_text,
            "has_images": len(image_filenames) > 0,
            "image_count": len(image_filenames),
            "image_filenames": image_filenames
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post('/analyze-single-report')
async def analyze_single_report(
    image: UploadFile = File(...),
    question: str = Form("Please analyze this feline CKD medical report and provide insights.")
):
    try:
        # Read and encode the image
        image_data = await image.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        
        # Determine image format
        content_type = image.content_type or "image/jpeg"
        data_url = f"data:{content_type};base64,{image_base64}"
        
        # Create multimodal message
        message = HumanMessage(content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": data_url}}
        ])
        
        # Process with the graph
        result = graph_builder.invoke({
            "messages": [message],
        })
        
        response_text = result["messages"][-1].content
        
        # Log the interaction
        log_file = "static/logs/chatbot_logs.json"
        log_entry = {
            "query": f"Single report analysis: {question}",
            "response": response_text,
            "image_filename": image.filename
        }
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "logs" not in data:
                    data["logs"] = []
        else:
            data = {"logs": []}

        data["logs"].append(log_entry)

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return JSONResponse(content={
            "query": question,
            "response": response_text,
            "image_filename": image.filename
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})