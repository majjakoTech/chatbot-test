from langgraph.graph import START, END, StateGraph
from dotenv import load_dotenv
from typing import Annotated, TypedDict
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

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)  # Slightly higher temperature for more natural responses

# Create empathetic prompt template
empathetic_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
🌟 You are a specialized AI assistant for feline Chronic Kidney Disease (CKD) information. All responses should be specifically about cats with kidney disease

📚 **Context Information:**
{context}

❓ **User's Question:**
{question}

🎯 **CRITICAL FORMATTING INSTRUCTIONS:**
- Always interpret the question in the context of feline CKD
- Format your response using STRUCTURED SECTIONS with bullet points
- Use ⸻ (em dash) to separate each section
- Start each section with an emoji and descriptive header
- Use bullet points (•) under each section header
- Be warm, empathetic, and understanding in your tone 💝
- Always emphasize the importance of veterinary consultation 🏥
- End with an encouraging note or offer to help further if needed ✨

📝 **REQUIRED RESPONSE FORMAT:**
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
🏥 [Veterinary Advice Section]:
• [Bullet point about consulting vet]
• [Bullet point about monitoring]
⸻
🐾 Bottom Line:
• [Key takeaway 1]
• [Key takeaway 2]
• [Encouraging closing statement]

💬 **Your structured, empathetic response (always in feline CKD context):**
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

# Enhanced chatbot node logic with empathetic fallback
def chatbot(state: State) -> State:
    last_human_msg = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
    query = last_human_msg.content
    
    # Get documents directly from retriever to check if any were found
    retrieved_docs = retriever.invoke(query)
    
    if not retrieved_docs:
        # More empathetic fallback message
        fallback = """
😔 I don't have specific information about that topic in my feline CKD knowledge base. 

However, I'd be happy to help you with CKD-related questions about:
- Diet and nutrition for cats with kidney disease
- Symptoms and monitoring
- Treatment options
- Care and management

What would you like to know about feline CKD? 💙
        """.strip()
        return {"messages": state["messages"] + [AIMessage(content=fallback)]}
    
    # If we have documents, proceed with QA chain
    response = qa_chain.invoke({"query": query})
    response_text = response["result"]
    docs = response.get("source_documents", [])
    
    # Optional logging
    # print("Retrieved documents:")
    # for i, doc in enumerate(docs):
    #     print(f"\n--- Document {i+1} ---")
    #     print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
    #     print(doc.metadata)
    
    return {"messages": state["messages"] + [AIMessage(content=response_text)]}


# Build the LangGraph
graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

graph_builder = graph.compile()

# Enhanced callable function for API or UI
def get_results(query: str, include_greeting: bool = False):
    
    result = graph_builder.invoke({
        "messages": [HumanMessage(content=query)],
    })
    log_file = "static/logs/chatbot_logs.json"

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