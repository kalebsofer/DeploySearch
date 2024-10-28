from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union

from .preprocess import preprocess
from .get_docs import get_docs

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class DocumentResponse(BaseModel):
    rel_docs: List[str]
    rel_docs_sim: List[Union[float, int]]

@app.get("/")
async def read_root():
    return {"message": "Welcome to the search API. Use POST /search to send queries."}


@app.post("/search", response_model=DocumentResponse)
async def search(query_request: QueryRequest):
    query = query_request.query
    # query_embedding = preprocess(query)
    rel_docs, distances  = get_docs(query)
    return {
        "rel_docs": rel_docs,
        # "urls": urls,
        "rel_docs_sim": distances[0]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
