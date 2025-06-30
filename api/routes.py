import os
import json
from typing import Optional
from dotenv import load_dotenv
from fastapi import Query,APIRouter
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse

from services.chatbot import get_results

load_dotenv()

router=APIRouter()

@router.get('/search')
async def search(q:str=Query(...,description="Search Query"), city_id: Optional[int] = Query(None, description="Optional City ID")):
    try:
        response = get_results(q)
        return JSONResponse(content={"query": q, "response": response})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})