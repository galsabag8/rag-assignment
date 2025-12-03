from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
import os

app = FastAPI()

SIMILARITY_THRESHOLD = 1.4

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

# --- LOAD DATABASE ONCE (Startup) ---
api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Input Schema
class QueryRequest(BaseModel):
    query: str

@app.post("/search")
def search(request: QueryRequest):
    try:
        # 1. Validation
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        # 2. Search
        results = db.similarity_search_with_score(request.query, k=3)
        
        response = []
        for doc, score in results:
            if score > SIMILARITY_THRESHOLD:
                continue 
                
            response.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": round(score, 4)
            })
            
        return response

    except Exception as e:
        # This catches API errors, DB errors, etc.
        print(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error: Something went wrong with the search engine.")

# To run this: uvicorn main:app --reload