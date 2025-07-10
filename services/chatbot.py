from langgraph.graph import START, END, StateGraph
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Union, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from qdrant_client import QdrantClient
from pydantic import BaseModel, Field
from typing_extensions import Literal
import os
import json
import base64
from PIL import Image
import io

# Load environment variables from .env (ensure OPENAI_API_KEY is set)
load_dotenv()

# Connect to Qdrant
client = QdrantClient(url="http://localhost:6333")

# Initialize OpenAI Embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Setup vector store
vector_store = Qdrant(
    client=client,
    collection_name="hugging_cat",
    embeddings=embedding_model
)

# Configure retriever with proper parameters
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.15,
        "k": 5
    }
)

# Initialize OpenAI LLM with vision capabilities
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# Create multimodal prompt template for multiple image analysis
multi_image_analysis_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
🌟 You are a specialized AI assistant for analyzing multiple feline Chronic Kidney Disease (CKD) reports and medical documents. You ONLY analyze medical reports related to cats with kidney disease.

📚 **Context Information (if available):**
{context}

❓ **User's Question:**
{question}

 **STRICT SCOPE LIMITATION:**
- ONLY analyze medical reports specifically related to feline CKD (Chronic Kidney Disease in cats)
- If the images are not medical reports or not related to feline kidney disease, you MUST respond that you can only analyze feline CKD reports
- Always interpret medical data in the context of feline CKD ONLY
- When analyzing multiple reports, compare and contrast findings across all reports

🎯 **CRITICAL FORMATTING INSTRUCTIONS:**
- Format your response using STRUCTURED SECTIONS with bullet points
- Use ⸻ (em dash) to separate each section
- Start each section with an emoji and descriptive header
- Use bullet points (•) under each section header
- Be warm, empathetic, and understanding in your tone 💝
- Always emphasize the importance of veterinary consultation 🏥
- End with an encouraging note or offer to help further if needed ✨

📝 **REQUIRED RESPONSE FORMAT:**
Start with a brief overview of what you see across all medical reports, then organize information into sections like this:

⸻
🔬 [Overall Medical Report Analysis]:
• [Key findings across all reports]
• [Important values and measurements from multiple reports]
• [Any concerning indicators found in any report]
⸻
📊 [Comparative Analysis]:
• [How values compare across different reports]
• [Trends or changes over time (if multiple reports from same cat)]
• [Consistencies or discrepancies between reports]
⸻
 [Individual Report Highlights]:
• [Specific findings from each report]
• [Unique aspects of each report]
• [How each report contributes to the overall picture]
⸻
⚠️ [Important Warnings]:
• [Any critical values that need attention]
• [Red flags to watch for across all reports]
• [Urgent veterinary concerns]
⸻
🏥 [Veterinary Recommendations]:
• [When to consult a veterinarian]
• [What to monitor based on all reports]
• [Next steps for care considering all findings]
⸻
🐾 Bottom Line:
• [Key takeaway 1]
• [Key takeaway 2]
• [Encouraging closing statement]

💬 **Your structured response:**
"""
)

# Create empathetic prompt template for text-only queries
empathetic_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
🌟 You are a specialized AI assistant EXCLUSIVELY for feline Chronic Kidney Disease (CKD) information. You ONLY answer questions about cats with kidney disease.

📚 **Context Information:**
{context}

❓ **User's Question:**
{question}

 **STRICT SCOPE LIMITATION:**
- ONLY answer questions specifically related to feline CKD (Chronic Kidney Disease in cats)
- If the question is about cats but NOT about kidney disease (e.g., cat breeds, general cat care, non-kidney health issues), you MUST respond that you don't have information on that topic
- If the context provided doesn't contain relevant information about the specific CKD-related question asked, say you don't have information on that aspect of feline CKD
- Always interpret questions in the context of feline CKD ONLY

🎯 **CRITICAL FORMATTING INSTRUCTIONS:**
- Format your response using STRUCTURED SECTIONS with bullet points
- Use ⸻ (em dash) to separate each section
- Start each section with an emoji and descriptive header
- Use bullet points (•) under each section header
- Be warm, empathetic, and understanding in your tone 💝
- Always emphasize the importance of veterinary consultation 🏥
- End with an encouraging note or offer to help further if needed ✨

📝 **REQUIRED RESPONSE FORMAT (ONLY if question is about feline CKD):**
Start with a brief overview sentence, then organize information into sections like this:

⸻
🔬 [Section Header with Emoji]:
• [Bullet point 1]
• [Bullet point 2]
• [Bullet point 3]
⸻
✅ [Another Section Header]:
• [Bullet point 1]
• [Bullet point 2]
⸻
❌ [Warning/Limitation Section if applicable]:
• [Bullet point 1]
• [Bullet point 2]
⸻
 [Veterinary Advice Section]:
• [Bullet point about consulting vet]
• [Bullet point about monitoring]
⸻
🐾 Bottom Line:
• [Key takeaway 1]
• [Key takeaway 2]
• [Encouraging closing statement]

💬 **Your structured response (ONLY if question is about feline CKD):**
"""
)

# RetrievalQA chain with custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": empathetic_prompt
    }
)

# State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Enhanced chatbot node logic with multiple image processing
def chatbot(state: State) -> State:
    last_human_msg = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
    
    # Check if the message contains images
    if isinstance(last_human_msg.content, list):
        # Handle multimodal input (text + multiple images)
        text_content = ""
        image_contents = []
        
        for content_block in last_human_msg.content:
            if isinstance(content_block, dict):
                if content_block.get("type") == "text":
                    text_content = content_block.get("text", "")
                elif content_block.get("type") == "image_url":
                    image_url = content_block.get("image_url", {}).get("url", "")
                    if image_url:
                        image_contents.append(image_url)
        
        # If we have images, use vision model directly
        if image_contents:
            # Create content blocks for multiple images
            content_blocks = [{"type": "text", "text": text_content or "Please analyze these feline CKD medical reports and provide comprehensive insights."}]
            
            for image_url in image_contents:
                content_blocks.append({
                    "type": "image_url", 
                    "image_url": {"url": image_url}
                })
            
            # Create a vision-capable message
            vision_message = HumanMessage(content=content_blocks)
            
            # Get response from vision model
            response = llm.invoke([vision_message])
            response_text = response.content
            
            # Add context from vector store if available
            if text_content:
                retrieved_docs = retriever.invoke(text_content)
                if retrieved_docs:
                    context = "\n".join([doc.page_content for doc in retrieved_docs])
                    # Use the multi-image analysis prompt with context
                    formatted_prompt = multi_image_analysis_prompt.format(
                        context=context,
                        question=text_content
                    )
                    # Create a new message with the formatted prompt
                    enhanced_content_blocks = [{"type": "text", "text": formatted_prompt}]
                    for image_url in image_contents:
                        enhanced_content_blocks.append({
                            "type": "image_url", 
                            "image_url": {"url": image_url}
                        })
                    enhanced_message = HumanMessage(content=enhanced_content_blocks)
                    enhanced_response = llm.invoke([enhanced_message])
                    response_text = enhanced_response.content
        else:
            # Handle text-only input (existing logic)
            query = text_content
            retrieved_docs = retriever.invoke(query)
            
            if not retrieved_docs:
                fallback = """
I don't have specific information about that topic in my feline CKD knowledge base.

⸻
😔 What I Specialize In:
• I'm specifically designed to help with feline Chronic Kidney Disease (CKD) information
• I can only provide information about cats with kidney disease
• For general cat questions or other health topics, I'm not able to help
⸻
🩺 Topics I Can Help With:
• Diet and nutrition for cats with kidney disease
• CKD symptoms and monitoring
• Treatment options and medications for feline CKD
• Care and management of cats with kidney disease
• Understanding test results related to kidney function
⸻
 Let's Focus on CKD:
• What would you like to know about feline Chronic Kidney Disease?
• I'm here to help with any kidney-related concerns for your cat 💙
                """.strip()
                return {"messages": state["messages"] + [AIMessage(content=fallback)]}
            
            response = qa_chain.invoke({"query": query})
            response_text = response["result"]
    else:
        # Handle text-only input (existing logic)
        query = last_human_msg.content
        retrieved_docs = retriever.invoke(query)
        
        if not retrieved_docs:
            fallback = """
I don't have specific information about that topic in my feline CKD knowledge base.

⸻
😔 What I Specialize In:
• I'm specifically designed to help with feline Chronic Kidney Disease (CKD) information
• I can only provide information about cats with kidney disease
• For general cat questions or other health topics, I'm not able to help
⸻
🩺 Topics I Can Help With:
• Diet and nutrition for cats with kidney disease
• CKD symptoms and monitoring
• Treatment options and medications for feline CKD
• Care and management of cats with kidney disease
• Understanding test results related to kidney function
⸻
 Let's Focus on CKD:
• What would you like to know about feline Chronic Kidney Disease?
• I'm here to help with any kidney-related concerns for your cat 💙
            """.strip()
            return {"messages": state["messages"] + [AIMessage(content=fallback)]}
        
        response = qa_chain.invoke({"query": query})
        response_text = response["result"]
    
    return {"messages": state["messages"] + [AIMessage(content=response_text)]}

# Build the LangGraph
graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

graph_builder = graph.compile()

# Enhanced callable function for API or UI
def get_results(query: str, include_greeting: bool = False):
    
    try:
        result = graph_builder.invoke({
            "messages": [HumanMessage(content=query)],
        })
        
        log_file = "static/logs/chatbot_logs.json"
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        log_entry = {
            "query": query,
            "response": result["messages"][-1].content
        }
        
        # Create a log entry
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "logs" not in data:
                    data["logs"] = []
        else:
            data = {"logs": []}

        # Append new entry
        data["logs"].append(log_entry)

        # Write back to file
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return result["messages"][-1].content
        
    except Exception as e:
        print(f"Error in get_results: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"