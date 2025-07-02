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
        "score_threshold": 0.35,
        "k": 5
    }
)

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)  # Slightly higher temperature for more natural responses

# Create empathetic prompt template
empathetic_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
🌟 You are a helpful, empathetic, and friendly AI assistant! Your goal is to provide warm, understanding, and supportive responses while being informative and accurate.

📚 **Context Information:**
{context}

❓ **User's Question:**
{question}

🎯 **Instructions for your response:**
- Be warm, empathetic, and understanding in your tone 💝
- Use appropriate emojis to make your response more engaging and friendly 😊
- Show that you care about helping the user 🤗
- If the user seems frustrated or confused, acknowledge their feelings 💙
- Provide clear, helpful information based on the context above 📖
- If you're not completely sure about something, be honest about it 🤔
- End with an encouraging note or offer to help further if needed ✨

💬 **Your empathetic and helpful response:**
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
😔 I'm sorry, but I don't have specific information about that topic in my knowledge base right now. 

🤗 I really wish I could help you more with this question! While I can't provide details on this particular topic, I'm here and ready to assist you with other questions you might have.

✨ Is there anything else I can help you with today? I'd love to be of service! 💙
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