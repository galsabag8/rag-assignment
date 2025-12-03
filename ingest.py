import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

# --- CONFIG ---
# Make sure you set your key! 
api_key = os.getenv("OPENAI_API_KEY")
def ingest_data():
    print("1. Loading CSV...")
    # Load the CSV using Pandas
    df = pd.read_csv("data.csv")
    
    documents = []
    
    # 2. Convert Rows to Text Chunks
    # We iterate through each book (row) and create a nice string description
    for index, row in df.iterrows():
        
        # Combine the columns into a single meaningful paragraph
        # This is what the LLM/Vector DB will actually "read"
        combined_text = (
            f"Title: {row['name']}\n"
            f"Author: {row['authors']}\n"
            f"Quote: {row['favorite_quote']}\n"
            f"Review: {row['One_line_review']}\n"
            f"Why Read: {row['why_should_read']}"
        )
        
        # Create a LangChain Document
        # We store the 'name' in metadata so we can display it nicely in the UI later
        doc = Document(
            page_content=combined_text,
            metadata={"title": row['name'], "author": row['authors']}
        )
        documents.append(doc)
    
    print(f"   Created {len(documents)} documents.")

    # 3. Embed and Store
    print("3. Embedding and Saving to ChromaDB...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # This creates the folder "./chroma_db" and saves everything there
    db = Chroma.from_documents(
        documents, 
        embeddings, 
        persist_directory="./chroma_db"
    )
    print("Success! Database created at './chroma_db'")

if __name__ == "__main__":
    ingest_data()