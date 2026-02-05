# GenAI RAG User Training Assistant - Sandoz x UC3M

**Generative AI training assistant built to help users navigate complex internal documentation through a chat UI, backed by a Hybrid Retrieval-Augmented Generation (RAG) System with traceable citations, feedback and topic-aware retrieval.**

For a deep dive into the complete development, design decisions, pipelines, technical specs, evaluation scores and challenges, refer to the **Final Report** included in the repository:  
📄 [`Final Report (PDF)`](./FINAL_REPORT_SANDOZ.pdf).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🎥 **Watch the Demo**: Check out a short video of the agent in action with 3 simple use cases, attached in my [LinkedIn Post](https://www.linkedin.com/posts/c%C3%A9sar-%C3%A1lvarez-cascos-herv%C3%ADas-176ab0281_ai-generativeai-rag-activity-7421532524469788672-mpRD?utm_source=share&utm_medium=member_desktop&rcm=ACoAAESvA-oBt01h6GKNRdKiesdRWKDay17085Q)

## 🚀 Overview

This project implements an intelligent conversational agent designed to help users navigate the **SANITY** platform (Sandoz's internal management system). Instead of manually searching through hundreds of PDF manuals, users can ask natural language questions and receive instant, accurate and cited answers grounded in the company's official documentation.  
  
The system features a **Hybrid Architecture** capable of running in **Cloud Mode** (OpenAI + Supabase) for maximum performance or **Local Mode** (LM Studio + Local Embeddings) for data privacy compliance.

## 🛠️ Technical Architecture

This project is an end-to-end RAG solution covering the entire data lifecycle:

#### 1. Data Ingestion & Processing (`ingest/`)

Smart pipeline to insert structured and indexed content into the database, avoiding duplicated files and detecting updated versions.

**PDF Parsing:** Uses Docling and RapidOCR to extract text and analyze layout from complex PDF reports.

**Chunking:** Semantic splitting based on page breaks and overlapping windows to preserve context.

**Embeddings:** Generates vector representations of chunks using text-embedding-3-small for the cloud pipeline.

#### 2. Semantic Structuring (`src/classification.py`)

**Topic Modeling:** Implements NMF (Non-Negative Matrix Factorization) combined with spaCy for lemmatization to automatically cluster documents into semantic topics.  
Organizes the corpus into coherent areas, making retrieval more targeted and interpretable.

**Auto-Labeling:** Uses LLMs to generate human-readable titles and descriptions for these clusters, improving retrieval precision, with the option to export and import the topics table for human refinement.

#### 3. Retrieval Pipeline (`src/search_kb.py`)

**Hybrid Search:** Combines Vector Similarity Search (Cosine distance) with Keyword Search (pg_trgm) using a weighted fusion algorithm.

**Topic Awareness:** Automatically detects the query's topic to filter the search space, reducing hallucinations. If the system cant't confidently infer a topic, the UI can prompt the user to pick one.

#### 4. Generation & Feedback (`api/`, `src/`)

Design to support continuous improvement (rating + optional comments).

**Adaptive Prompting:** The system learns from user feedback (stored in Supabase). If a query matches previous positive interactions, it injects "few-shot" examples into the prompt to guide the LLM.

**Citations:** Enforces strict evidence citation [n] to ensure all answers are grounded in the retrieved documents.

#### 5. Chat UI (`chat.html`)

Clean, responsive frontend interface for interacting with the agent.  
Displays answers plus expandable sources, clickable citations for traceability and UX.  
Allows language selection (ENG/SP).

#### 6. SQL Schemas (`schema/`)

SQL scripts used to initialize the database structure in Supabase (tables for documents, chunks, topics, and feedback).

## ⚡Run the Agent

### Prerequisites & External Services
To run this project as configured, you will need:
* **Supabase Project:** Used as the Vector Database with the corresponding tables (PostgreSQL with `pgvector`).
* **OpenAI API Key:** Used for embeddings and LLM generation.

> **Note:** While the codebase is currently configured for these services, the modular architecture allows for adaptation to other LLM providers (e.g., Anthropic, HuggingFace) or cloud databases with appropriate code modifications.

### 1) Clone the repository & Install dependencies
```bash
git clone https://github.com/cesarAlvarezCascos/GenAI_RAG_Training_Assistant
cd GenAI_RAG_Training_Assistant
pip install -r requirements.txt
```
### 2) Download spaCy language model required for NLP and Topic Modeling tasks
```bash
python -m spacy download en_core_web_lg
```
### 3) Configure environment variables

Rename the provided example file `.env-example` to `.env` and fill in your credentials.
```bash
mv .env-example .env
```
Ensure your `.env` file contains the following keys:  
API keys (if applicable)  
Database/Supabase URL and credentials  
Any model endpoints / embedding configuration 

### 4) Ingest Documentation

To work, the agent needs a document corpus. The current pipeline is configured to ingest PDFs directly from the **Supabase Storage Bucket**, although the code can be adapted to ingest from a local folder or other cloud providers.  

To process the documents and fill the database, run:  
```bash
python ingest/ingest_pdfs.py
```

### 5) Run the agent in your local machine

Open two separate terminal windows.  
  
**Step 1:** Start the API. Launch backend on port 8000  
```bash
uvicorn api.main:app --reload --port 8000
```  
  
**Step 2:** Start the Frontend. Serve the UI on port 5500  
From the repository root:  
```bash
python -m http.server 5500
```  
  
**Step 3:** Acces the App. Open in your browser:  
[http://localhost:5500/chat.html](http://localhost:5500/chat.html)  
<br>

## 🛡️ License & Credits
Developed as part of the Data Science & Engineering Degree at Universidad Carlos III de Madrid in collaboration with Sandoz.

See [`FINAL REPORT SANDOZ.pdf`](./FINAL_REPORT_SANDOZ.pdf) for full team credits.
