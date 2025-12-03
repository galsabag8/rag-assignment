# 📚 Book Wisdom Search: RAG Retrieval System

## 📌 Overview
This project is a Retrieval-Augmented Generation (RAG) pipeline designed to ingest a structured dataset of book reviews, generate vector embeddings, and serve relevant context via a semantic search API.

The system allows users to ask abstract questions (e.g., *"Why is motivation weird?"*) and retrieves the most semantically relevant book quotes and reviews from a local vector database, demonstrating the power of semantic search over keyword matching.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
* **Python 3.10+**
* **An OpenAI API Key** (Required for generating embeddings)

### 2. Installation
Clone the repository and set up the isolated environment:

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate environment
# Windows (PowerShell):
.\venv\Scripts\Activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 3.  Configuration 
Set your OpenAI API key as an environment variable. This prevents hardcoding secrets in the codebase and follows security best practices.

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-your-key-here"

# Mac/Linux / Git Bash
export OPENAI_API_KEY="sk-your-key-here"
```

### 4. Ingestion (Run Once)
Process the raw CSV data and populate the Vector Database:

```bash
python ingest.py
```
Expected Output: Success! Database created at './chroma_db'

### 5.Running the Application
Start the FastAPI backend server:

```bash
uvicorn main:app --reload
```
### 6. Using the Search UI
Simply double-click the `index.html` file in the project folder to open it in your browser.

Once the interface loads, enter a natural language query (e.g., *"Why is motivation weird?"*) into the input field and click **Search** to retrieve and display the most relevant context blocks.

Alternatively, verify the API backend directly at: `http://127.0.0.1:8000/docs`


### 🧠 Architectural Decisions & Reasoning

### 1. Dataset Selection
* **Dataset:** *Bestseller Happiness Books Reviews* (Source: Kaggle).
* **Why this dataset?**
  The assignment required a small, structured text dataset. I chose this specific dataset because it contains rich semantic fields (`One_line_review`, `favorite_quote`, `why_should_read`) rather than dry statistics.
* **Fit for RAG:**
  Unlike factual datasets (e.g., "Product Specs"), this dataset allows for **abstract reasoning queries**. Users can ask about feelings or concepts ("finding purpose") rather than just keywords, which effectively demonstrates the superiority of Vector Embeddings.
* **Expected User Questions:**
  Since the content focuses on psychology and self-improvement, the system is designed to handle:
  1. **Abstract Concepts:** "Why is motivation weird?" (Semantic matching).
  2. **Specific Retrieval:** "What book uses the elephant and rider metaphor?" (Fact retrieval).
  3. **Thematic Discovery:** "Books about the science of habits" (Broad categorization).

### 2. Vector Database: ChromaDB
I selected ChromaDB over cloud-based options (Pinecone) or heavy infrastructure (pgvector) for three reasons:

* **Local Persistence:** Chroma runs entirely on the local file system (./chroma_db). This eliminates network latency during retrieval and removes the need for external API keys or cloud account management for the reviewer.

* **Simplicity:** It requires no Docker containers, making the "Time to Hello World" extremely fast for a home assignment scope.

* **LangChain Integration:** It has first-class support in the LangChain ecosystem, allowing for clean, readable code.

### 3. Ingestion Strategy: "Logical Chunking"
* **Strategy:** Instead of using a fixed-size splitter (e.g., `RecursiveCharacterTextSplitter`), I implemented **Logical Chunking**. I programmatically combined relevant columns (`Title`, `Author`, `Quote`, `Review`) into a single, coherent text block per book.
* **Chunk Size:** **Dynamic**. The size is determined by the natural length of the book entry (typically 300-800 characters). This ensures that no review is arbitrarily cut off in the middle of a sentence.
* **Overlap:** **0 (None)**. Since each row represents a distinct book, adding overlap would blend information from two different authors, leading to factual errors. By keeping overlap at 0, every retrieved chunk is a self-contained, accurate unit of context including the Book Title and Author.

### 4. Retrieval Parameters (Top-K & Thresholding)
* **Top-K = 3:** During testing with queries like *"What book uses the elephant and rider metaphor?"*, the correct result sometimes appeared at rank #2. Setting `k=3` ensures the correct context is captured even if it isn't the absolute top match.
* **Distance Threshold (1.4):** I implemented a filter to reject results with a distance score > 1.4.
* **Reasoning:** ChromaDB uses Euclidean distance (where lower is better).
    * **Calibration Process:** I performed manual fine-tuning by running both relevant queries (which scored ~0.5 to ~1.4) and irrelevant queries like *"How to bake a cake"* (which scored >1.7).
    * **Outcome:** Setting the threshold at 1.4 proved optimal for catching abstract semantic matches while successfully rejecting noise to prevent hallucination-inducing context.

### 5. Frontend Architecture (Vanilla HTML/JS)
I chose a plain HTML/JavaScript implementation over a framework like React.

* **Reasoning:** For a single-page retrieval interface, a build step (Webpack/Vite) adds unnecessary complexity. This approach keeps the codebase lightweight, instantly testable, and dependency-free on the client side.

📂 Project Structure

├── chroma_db/          # The persistent Vector Database (Generated)
├── data.csv            # The raw dataset
├── ingest.py           # ETL pipeline: Loads CSV -> Embeds -> Stores in DB
├── main.py             # FastAPI Backend with Search Logic & CORS
├── index.html          # Frontend UI
├── requirements.txt    # Python dependencies
└── README.md           # Documentation

🧪 Example Test Cases

Query,Expected Result (Semantic Match)
"""Why is motivation weird?""", "The Happiness Equation (Quote: ""Action causes motivation"")"
"""Elephant and rider metaphor""", "The Happiness Hypothesis (Matches ""Why Read"" section)"
"""Books about science""", "Returns multiple results (Advantage, Hypothesis, Equation)"



