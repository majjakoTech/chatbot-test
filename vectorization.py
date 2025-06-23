import json
import uuid
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client.http.models import PointStruct,VectorParams,Distance

from db.Qdrant import qdrant_client

load_dotenv()

collection_name="hugging_cat"
points=[]

qdrant_client.recreate_collection(collection_name=collection_name,vectors_config=VectorParams(size=3072,distance=Distance.COSINE))
embeddings=OpenAIEmbeddings(model="text-embedding-3-large")

with open("documents/document.json","r") as file:
    data=json.load(file)
    for count,point in enumerate(data):
        
        vector_text=point["content"]+" ".join(point["metadata"]["potential_questions"])
        vector=embeddings.embed_query(vector_text)
        payload={
            "chunk_index":count,
            "content":point["content"],
            "metadata":{
                **point["metadata"],
                "preview":point["content"][:3000]
                }
        }
        qdrant_client.upsert(collection_name=collection_name,points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )])
        print("Point Vectorized and inserted",count+1)


print("!! VECTORIZATION SUCCESSFULL !!")