from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union
from minio import Minio

from .search_engine import SearchEngine
from .config.settings import get_settings

settings = get_settings()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In prod, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class DocumentResponse(BaseModel):
    rel_docs: List[str]
    rel_docs_sim: List[Union[float, int]]


# Initialize search engine on startup
search_engine = None


@app.on_event("startup")
async def startup_event():
    global search_engine
    try:
        search_engine = SearchEngine()
    except Exception as e:
        print(f"Error initializing search engine: {e}")
        raise

    # Use settings for MinIO configuration
    client = Minio(
        settings.MINIO_URL,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )


@app.get("/")
async def read_root():
    return {"message": "Welcome to the search API. Use POST /search to send queries."}


@app.post("/search", response_model=DocumentResponse)
async def search(query_request: QueryRequest):
    try:
        if not search_engine:
            raise HTTPException(status_code=503, detail="Search engine not initialized")

        rel_docs, distances = search_engine.search(query_request.query)
        return {"rel_docs": rel_docs, "rel_docs_sim": distances[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
