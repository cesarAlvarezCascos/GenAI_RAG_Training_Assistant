# src/search_kb_local.py
import os
import psycopg
from dotenv import load_dotenv
from typing import List, Dict, Any

from src.local_embedding import LocalEmbeddingPipeline

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
conn = psycopg.connect(DATABASE_URL)

_emb_pipe = LocalEmbeddingPipeline()


def _embed_query_local(q: str) -> list[float]:
    return _emb_pipe.embed_text(q)


def vector_search_local(qvec: list[float], k: int = 8) -> List[Dict[str, Any]]:
    """
    Búsqueda vectorial sobre las tablas local_* usando pgvector.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 
                c.id AS chunk_id,
                c.body AS chunk_text,
                d.id AS document_id,
                d.file_name,
                c.page_number,
                d.pdf_url,
                1 - (ce.embedding <=> %s::vector) AS vscore
            FROM local_chunk_embeddings ce
            JOIN local_chunks c ON ce.chunk_id = c.id
            JOIN local_documents d ON c.document_id = d.id
            ORDER BY ce.embedding <=> %s::vector
            LIMIT %s;
            """,
            (qvec, qvec, k),
        )
        rows = cur.fetchall()

    results = []
    for (
        chunk_id,
        chunk_text,
        document_id,
        file_name,
        page_number,
        pdf_url,
        vscore,
    ) in rows:
        results.append(
            {
                "chunk_id": str(chunk_id),
                "snippet": chunk_text[:600],
                "document_id": str(document_id),
                "file_name": file_name,
                "page_number": page_number,
                "pdf_url": pdf_url,
                "score": float(vscore),
            }
        )
    return results


def search_kb_local(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """
    Entrada de alto nivel para el RAG local.
    """
    if not query or not query.strip():
        return []
    qvec = _embed_query_local(query)
    return vector_search_local(qvec, k=top_k)
