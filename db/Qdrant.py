import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST=os.environ["QDRANT_HOST"]
QDRANT_PORT=int(os.environ["QDRANT_PORT"])

qdrant_client=QdrantClient(host=QDRANT_HOST,port=QDRANT_PORT,timeout=120.0)
