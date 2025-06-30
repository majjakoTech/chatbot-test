from langgraph.graph import START, END, StateGraph
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from pydantic import BaseModel,Field
from typing_extensions import Literal
import os

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

# Retriever + OpenAI LLM
retriever = vector_store.as_retriever()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True )

# State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]
    decision:str

class CatRelated(BaseModel):
    step:Literal["yes","no"]=Field(description="CKD related or not")


# Chatbot node logic
def chatbot(state: State) -> State:
    last_human_msg = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
    query = last_human_msg.content

    response = qa_chain.invoke({"query": query})
    response_text = response["result"]

    # docs = response.get("source_documents", [])

    # # Optional: log or print retrieved docs
    # print("Retrieved documents:")
    # for i, doc in enumerate(docs):
    #     print(f"\n--- Document {i+1} ---")
    #     print(doc.page_content)
    #     print(doc.metadata)

    return {"messages": state["messages"] + [AIMessage(content=response_text)]}

router=llm.with_structured_output(CatRelated)

def llm_call_router(state:State):
    """ Route the input to the appropriate node"""
   
    decision=router.invoke([
        SystemMessage(
            content="""You are a strict classifier. Return 'yes' if the following query is related to cats,cat health, feline chronic kidney disease (CKD), cat nutrition, or general feline care. Otherwise, return 'no'. Respond with only 'yes' or 'no'."""
        ),
        HumanMessage(content=state["messages"][-1].content)
    ])

    return {"decision":decision.step}

def route_decision(state:State):

    if state["decision"]=="yes":
        return "chatbot"
    elif state["decision"]=="no":
        return "end"

# Build the LangGraph
graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_node("llm_call_router", llm_call_router)

graph.add_edge(START,"llm_call_router")
graph.add_conditional_edges("llm_call_router",route_decision,{
    "chatbot":"chatbot",
    "end":END
})

graph_builder = graph.compile()

def get_results(query: str):

    result = graph_builder.invoke({
        "messages": [HumanMessage(content=query)],
    })

    if result["decision"]=='yes':
        return result["messages"][-1].content
    else:
        return "Sorry this is out of scope for me!"
