# ingest/ingest_local_simple.py

import os
import glob
import hashlib
import shutil
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from supabase import create_client
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import sys

# Añadir la carpeta raíz del proyecto al sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Embeddings locales
from src.local_embedding import LocalEmbeddingPipeline

load_dotenv()

# Conexión a Postgres (Supabase)
DATABASE_URL = os.getenv("DATABASE_URL")
conn = psycopg.connect(DATABASE_URL)

# Supabase Storage (si quieres seguir subiendo PDFs al bucket "pdfs")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET_NAME = "pdfs-local"


# ===== Helpers Supabase Storage =====

def upload_pdf_to_supabase(pdf_path: str) -> str | None:
    file_name = os.path.basename(pdf_path)
    with open(pdf_path, "rb") as f:
        data = f.read()

    try:
        supabase.storage.from_(BUCKET_NAME).upload(
            path=file_name,
            file=data,
            file_options={
                "cache-control": "3600",
                "upsert": "true",
                "content-type": "application/pdf",
            },
        )
    except Exception as e:
        print(f"[ERROR] Failed to upload {file_name} to Supabase: {e}")
        return None

    try:
        url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_name)
        return url
    except Exception as e:
        print(f"[ERROR] Failed to get public URL for {file_name}: {e}")
        return None


def delete_from_supabase(file_name: str):
    if not file_name:
        return
    try:
        res = supabase.storage.from_(BUCKET_NAME).remove([file_name])
        if res.get("error"):
            print(f"[ERROR] Could not delete {file_name} from Supabase: {res['error']}")
        else:
            print(f"[DELETE] {file_name} removed from Supabase")
    except Exception as e:
        print(f"[ERROR] Failed to delete {file_name} from Supabase: {e}")


# ===== HASH / DUPLICADOS (tablas LOCAL) =====

def compute_pdf_hash(pdf_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def hash_exists_local(content_hash: str):
    """
    Verifica si ya existe un doc con ese hash en local_documents.
    Devuelve (id, file_name) o None.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, file_name FROM local_documents WHERE content_hash = %s",
            (content_hash,),
        )
        return cur.fetchone()


def archive_old_version(pdf_path: str) -> str:
    """
    Mueve el PDF viejo a ./archived/ en disco local (no en Supabase).
    """
    archive_dir = os.path.join(os.path.dirname(pdf_path), "archived")
    os.makedirs(archive_dir, exist_ok=True)

    dest = os.path.join(archive_dir, os.path.basename(pdf_path))
    if os.path.exists(dest):
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(os.path.basename(pdf_path))
        dest = os.path.join(archive_dir, f"{name}_{ts}{ext}")

    shutil.move(pdf_path, dest)
    return dest


def find_similar_local_document(embedding_avg: list[float], file_name: str, threshold=0.95):
    """
    Busca documentos locales con embedding_avg muy similar (pgvector).
    Devuelve (doc_id, file_name, similarity) del más parecido o None.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 
                id,
                file_name,
                1 - (embedding_avg <=> %s::vector) AS sim_embedding,
                similarity(file_name, %s) AS sim_filename
            FROM local_documents
            WHERE embedding_avg IS NOT NULL
            ORDER BY sim_embedding DESC
            LIMIT 1;
            """,
            (embedding_avg, file_name),
        )

        rows = cur.fetchall()
        for doc_id, old_name, sim_emb, sim_fname in rows:
            if sim_emb >= threshold and sim_fname > 0.5:
                return (doc_id, old_name, sim_emb)
        return None


def delete_local_document_cascade(doc_id, pdf_path: str | None = None):
    """
    Borra documento local + chunks + embeddings.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM local_chunk_embeddings
            WHERE chunk_id IN (
                SELECT id FROM local_chunks WHERE document_id = %s
            );
            """,
            (doc_id,),
        )
        cur.execute("DELETE FROM local_chunks WHERE document_id = %s;", (doc_id,))
        cur.execute("DELETE FROM local_documents WHERE id = %s;", (doc_id,))
    conn.commit()

    if pdf_path and os.path.exists(pdf_path):
        archived_path = archive_old_version(pdf_path)
        tqdm.write(f"[ARCHIVE] Moved old file to: {archived_path}")

    if pdf_path:
        delete_from_supabase(os.path.basename(pdf_path))


# ===== DB helpers (tablas LOCAL) =====

def upsert_local_document(file_name, body, content_hash, embedding_avg, pdf_url):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO local_documents (file_name, body, content_hash, embedding_avg, pdf_url)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (file_name, body, content_hash, embedding_avg, pdf_url),
        )
        doc_id = cur.fetchone()[0]
    conn.commit()
    return doc_id


def upsert_local_chunk(document_id, idx, content, page_number):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO local_chunks (document_id, chunk_idx, body, page_number)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
            """,
            (document_id, idx, content, page_number),
        )
        chunk_id = cur.fetchone()[0]
    conn.commit()
    return chunk_id


def insert_local_chunk_embedding(chunk_id, embedding):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO local_chunk_embeddings (chunk_id, embedding)
            VALUES (%s, %s);
            """,
            (chunk_id, embedding),
        )
    conn.commit()


def sync_local_deletions(pdf_files: list[str]) -> int:
    """
    Elimina de las tablas local_* documentos cuyos PDFs ya no existan en la carpeta.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT id, file_name FROM local_documents;")
        db_files = cur.fetchall()

    existing_names = {Path(p).name for p in pdf_files}
    deleted_count = 0

    for doc_id, file_name in db_files:
        if file_name not in existing_names:
            tqdm.write(f"[SYNC DELETE] Removing '{file_name}' from local_* (file no longer exists)")
            delete_local_document_cascade(doc_id)
            deleted_count += 1

    return deleted_count


# ===== INGESTA LOCAL =====

def ingest_local_documents(data_dir: str):
    pdf_files = glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True)
    print(f"[INFO] Found {len(pdf_files)} PDF files in directory (local mode).")

    if not pdf_files:
        print("[INFO] No PDF files to process.")
        deleted = sync_local_deletions(pdf_files)
        print(f"\n{'=' * 60}")
        print("Ingestion Complete (local):")
        print(f"  🗑️  Files deleted from local_* DB: {deleted}")
        print(f"{'=' * 60}")
        return

    uploaded = 0
    skipped_identical = 0
    updated = 0

    # Embeddings locales
    emb_pipe = LocalEmbeddingPipeline()

    # Text splitter simple
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs (local)", unit="file"):
        file_name = Path(pdf_path).name

        # 1: hash
        content_hash = compute_pdf_hash(pdf_path)

        # 2: duplicado exacto
        existing = hash_exists_local(content_hash)
        if existing:
            tqdm.write(f"[SKIP] '{file_name}' is exact duplicate of '{existing[1]}' (local)")
            skipped_identical += 1
            continue

        # 3: cargar PDF
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
        except Exception as e:
            tqdm.write(f"[ERROR] Failed to load '{file_name}': {e}")
            continue

        # 4: chunking
        chunks = splitter.split_documents(documents)
        if not chunks:
            tqdm.write(f"[WARN] No chunks extracted from '{file_name}'")
            continue

        texts = [c.page_content for c in chunks]

        # 5: embeddings locales por chunk
        embeddings = emb_pipe.embed_texts(texts)

        # 6: embedding medio del doc
        embedding_avg = emb_pipe.average_embeddings(embeddings)

        # 7: detectar versión actualizada
        similar = find_similar_local_document(embedding_avg, file_name, threshold=0.95)
        if similar:
            old_id, old_name, similarity = similar
            tqdm.write(
                f"[UPDATE] '{file_name}' is updated version of '{old_name}' (sim: {similarity:.3f})"
            )

            old_pdf_path = None
            for pdf in pdf_files:
                if Path(pdf).name == old_name:
                    old_pdf_path = pdf
                    break

            tqdm.write(f"[DELETE] Removing old version '{old_name}' (local)...")
            delete_local_document_cascade(old_id, old_pdf_path)
            updated += 1

        # 8: insertar documento
        body = "\n".join(texts)
        pdf_url = upload_pdf_to_supabase(pdf_path)  # opcional; sigue usando el bucket "pdfs"

        doc_id = upsert_local_document(file_name, body, content_hash, embedding_avg, pdf_url)

        # 9: insertar chunks + embeddings
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            page_number = chunk.metadata.get("page", None)
            chunk_id = upsert_local_chunk(doc_id, idx, chunk.page_content, page_number)
            insert_local_chunk_embedding(chunk_id, emb)

        tqdm.write(f"[DONE] Uploaded '{file_name}' with {len(chunks)} chunks (local)")
        uploaded += 1

    # 10: sync deletions
    print("\n[INFO] Checking for deleted files (local)...")
    deleted = sync_local_deletions(pdf_files)

    print(f"\n{'=' * 60}")
    print("Ingestion Complete (local):")
    print(f"  ✅ New files uploaded: {uploaded}")
    print(f"  🔄 Files updated: {updated}")
    print(f"  ⏭️  Exact duplicates skipped: {skipped_identical}")
    print(f"  🗑️  Files deleted from local_* DB: {deleted}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    base_dir = os.getenv(
        "LOCAL_PDF_FOLDER_PATH",
        os.path.join(os.path.dirname(__file__), "..", "pdfs-locales"),
    )
    ingest_local_documents(base_dir)
