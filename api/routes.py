import os
import json
import base64
from typing import Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from fastapi import Query, APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from langchain_core.messages import HumanMessage, SystemMessage

from services.chatbot import get_results, graph_builder

load_dotenv()

router = APIRouter()

def load_tuning_parameters():
    """Load tuning parameters from JSON file"""
    try:
        with open("documents/tuning.json", "r", encoding="utf-8") as f:
            parameters = json.load(f)
        return parameters
    except Exception as e:
        # Return default parameters if file doesn't exist or has issues
        return {
            "model": "gpt-4o",
            "temperature": 0.9,
            "top_p": 1.0,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.3,
            "max_tokens": 4000,
            "stop": None,
            "logit_bias": {},
            "system_prompt": "You are Hugging Cat, a helpful and compassionate veterinary assistant focused on feline chronic kidney disease (CKD). You provide expert information in a warm, structured, and supportive tone.\n\nYour role is to:\n1. Provide helpful, accurate information about feline CKD, symptoms, management, and care\n2. Offer emotional support to cat owners dealing with their pet's health challenges\n3. Share practical tips for caring for cats with kidney disease\n4. Explain medical concepts in an accessible, caring manner\n5. Emphasize the importance of veterinary consultation for all medical decisions\n6. Explain topics in great detail, including related aspects like symptoms, diagnosis, diet, supplements, treatment options, etc.\n7. Use tables when they can enhance clarity\n\n⸻\n\nWhen responding:\n- Format your response using clearly structured sections with bullet points\n- Start each section with an emoji and a descriptive header\n- Use • bullets under each section\n- Be warm, empathetic, and understanding 💝\n- Always remind users to consult their vet 🏥\n- End with an encouraging note or offer to help further ✨\n- Use Markdown formatting when helpful (e.g., bold, tables, headers, emojis)\n\n⸻\n\nScope Limitation:\n- Only focus on feline CKD. If the question is unrelated (e.g., other cat health issues, breeds), explain kindly that your focus is CKD.\n- If the provided context does not contain relevant CKD information, say so transparently\n\nYou should be informative but never provide medical diagnoses or treatment plans. Always encourage veterinary guidance."
        }

def create_llm_with_parameters():
    """Create ChatOpenAI instance with current tuning parameters"""
    params = load_tuning_parameters()
    
    # Build parameters that ChatOpenAI accepts
    llm_params = {
        "model": params.get("model", "gpt-4o"),
        "temperature": params.get("temperature", 0.9),
        "top_p": params.get("top_p", 1.0),
        "presence_penalty": params.get("presence_penalty", 0.3),
        "frequency_penalty": params.get("frequency_penalty", 0.3),
        "max_tokens": params.get("max_tokens", 4000)
    }
    
    # Add stop sequences if provided
    stop_sequences = params.get("stop")
    if stop_sequences and len(stop_sequences) > 0:
        llm_params["stop"] = stop_sequences
    
    # Add logit bias if provided
    logit_bias = params.get("logit_bias")
    if logit_bias and len(logit_bias) > 0:
        llm_params["logit_bias"] = logit_bias
    
    return ChatOpenAI(**llm_params)

@router.get("/tuning", response_class=HTMLResponse)
async def tuning_page():
    """Serve the tuning page"""
    with open("static/tuning.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@router.get("/tuning/parameters")
async def get_tuning_parameters():
    """Get current tuning parameters"""
    try:
        with open("documents/tuning.json", "r", encoding="utf-8") as f:
            parameters = json.load(f)
        return JSONResponse(content=parameters)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/tuning/parameters")
async def update_tuning_parameters(
    temperature: float = Form(...),
    top_p: float = Form(...),
    frequency_penalty: float = Form(...),
    presence_penalty: float = Form(...),
    max_tokens: int = Form(...),
    stop: Optional[str] = Form(""),
    logit_bias: Optional[str] = Form("{}"),
    system_prompt: str = Form(...)
):
    """Update tuning parameters"""
    try:
        # Process stop sequences
        stop_sequences = None
        if stop and stop.strip():
            stop_sequences = [seq.strip() for seq in stop.split(',') if seq.strip()]
        
        # Process logit bias
        logit_bias_dict = {}
        if logit_bias and logit_bias.strip() and logit_bias.strip() != "{}":
            try:
                logit_bias_dict = json.loads(logit_bias)
            except json.JSONDecodeError:
                return JSONResponse(status_code=400, content={"error": "Invalid JSON format for logit_bias"})
        
        parameters = {
            "model": "gpt-4o",
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "stop": stop_sequences,
            "logit_bias": logit_bias_dict,
            "system_prompt": system_prompt
        }
        
        with open("documents/tuning.json", "w", encoding="utf-8") as f:
            json.dump(parameters, f, indent=2)
        
        return JSONResponse(content={"message": "Parameters updated successfully", "parameters": parameters})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/chat") 
async def chat(
    message: str = Form(...),
    images: Optional[List[UploadFile]] = File(None)
):
    try:
        # Load tuning parameters to get system prompt
        tuning_params = load_tuning_parameters()
        system_prompt = tuning_params.get("system_prompt", "You are a helpful assistant.")
        
        # Create LLM instance with current tuning parameters
        llm = create_llm_with_parameters()

        # Handle images if provided (read them before creating the generator)
        if images:
            # Process multiple images
            content_blocks = [{"type": "text", "text": message}]
            
            for image in images:
                # Read and encode the image
                image_data = await image.read()
                image_base64 = base64.b64encode(image_data).decode("utf-8")
                
                # Determine image format
                content_type = image.content_type or "image/jpeg"
                data_url = f"data:{content_type};base64,{image_base64}"
                
                content_blocks.append({
                    "type": "image_url", 
                    "image_url": {"url": data_url}
                })
            
            # Create multimodal message
            human_message = HumanMessage(content=content_blocks)
        else:
            # Text-only message
            human_message = HumanMessage(content=message)
        
        # Create messages with system prompt
        messages = [
            SystemMessage(content=system_prompt),
            human_message
        ]

        async def generate_response():
            response_text = ""
            
            # Stream the response from the LLM
            async for chunk in llm.astream(messages):
                if chunk.content:
                    response_text += chunk.content
                    # Send each chunk as Server-Sent Events format
                    yield f"data: {json.dumps({'token': chunk.content})}\n\n"
            
            # Log the interaction after streaming is complete
            log_file = "static/logs/chatbot_logs.json"
            
            # Prepare log entry
            if images:
                image_filenames = [image.filename for image in images]
                log_entry = {
                    "query": message,
                    "response": response_text,
                    "image_count": len(image_filenames),
                    "image_filenames": image_filenames
                }
            else:
                log_entry = {
                    "query": message,
                    "response": response_text
                }
            
            # Create logs directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # Read existing logs or create new structure
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "logs" not in data:
                        data["logs"] = []
            else:
                data = {"logs": []}

            # Add new log entry
            data["logs"].append(log_entry)

            # Write back to file
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Send end signal
            yield f"data: {json.dumps({'done': True})}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

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