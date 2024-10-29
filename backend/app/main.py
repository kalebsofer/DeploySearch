from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .search_engine import SearchEngine

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search engine
search_engine = None


class SearchQuery(BaseModel):
    query: str


@app.on_event("startup")
async def startup_event():
    global search_engine
    search_engine = SearchEngine()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/search")
async def search(query: SearchQuery) -> dict:
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    try:
        documents, similarities = search_engine.search(query.query)
        return {
            "rel_docs": documents,
            "rel_docs_sim": similarities[0] if similarities else [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test")
async def test_endpoint():
    return {
        "message": "Backend is reachable",
        "environment": {"host": "0.0.0.0", "port": 8051},
    }
