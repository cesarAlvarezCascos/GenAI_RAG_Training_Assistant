import sys
import os
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# Add project root to sys.path so 'src' can be imported -- NOT USED
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.embedding import EmbeddingPipeline
from src.classification import init_topic_classifier_from_db, assign_topics_to_chunks
from ingest.DBops import DatabaseOperator
from ingest.docling_loader import CustomDocumentLoader

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


# INGESTION PIPELINE  (reading PDFs files from Supabase Storage, not from Local)
def ingest_documents():

    operator = DatabaseOperator(SUPABASE_URL, SUPABASE_KEY)
    print(f"[INFO] Found {len(operator.pdf_files)} PDF files in Storage bucket '{operator.bucket}'.")

    if not operator.pdf_files:
        print("[INFO] No PDF files to process.")
        deleted = operator.sync_deletions()
        print(f"\n{'='*60}")
        print(f"Ingestion Complete:")
        print(f"  🗑️  Files deleted from DB: {deleted}")
        print(f"{'='*60}")
        return
    
    # COUNTERS:
    uploaded = 0
    skipped_identical = 0
    updated = 0

    # Initialize pipeline
    loader = CustomDocumentLoader()
    emb_pipe = EmbeddingPipeline(client)

    # Create temp directory
    temp_dir = "./temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
   

    # Process each doc individually
    for file_name in tqdm(operator.pdf_files, desc="Processing PDFs", unit="file"):
        # Download PDF temporarily, so we can compute hashes, embeddings, etc (our machine needs those files)
        pdf_path = operator.download_from_supabase(file_name, temp_dir)

        if not pdf_path:
            tqdm.write(f"[ERROR] Could not download '{file_name}'")
            continue

        # 1: Compute hash of content and verify if existing duplicate
        content_hash = operator.compute_pdf_hash(pdf_path)
        existing = operator.hash_exists(content_hash)
        if existing:
            tqdm.write(f"[SKIP] '{file_name}' is exact duplicate of '{existing[1]}'")
            skipped_identical += 1
            os.remove(pdf_path)  # Clean temp
            continue

        # 2: Load and process PDF file
        try:
            document_text = loader.process_document(pdf_path=pdf_path)
        except Exception as e:
            tqdm.write(f"[ERROR] Failed to load '{file_name}': {e}")
            continue

        # 3: Chunk the document
        chunks = loader.chunk_text(document_text, page_break_placeholder="[PAGE_BREAK]")

        # 4: Embeddings
        embeddings = emb_pipe.embed_chunks(chunks)
        if hasattr(embeddings, "data"):
            embeddings = [d.embedding for d in embeddings.data]

        # 5: Compute avg embedding of the doc
        embedding_avg = np.mean(embeddings, axis = 0).tolist()

        # 6: Look for a similar document
        similar = operator.find_similar_document(embedding_avg, file_name, threshold = 0.95)

        if similar:
            # It is an UPDATED DOC
            old_id, old_name, similarity = similar
            tqdm.write(f"[UPDATE] '{file_name}' is updated version of '{old_name}' (similarity: {similarity:.3f})")

            tqdm.write(f"[DELETE] Removing old version '{old_name}'...")
            operator.delete_document_cascade(old_id, old_name, should_archive=True)            
            updated += 1

        # 7: Insert New / Updated Document
        body = "\n".join([c["content"] for c in chunks])
        # URL that points to the document in the bucket Storage
        pdf_url = f"{SUPABASE_URL}/storage/v1/object/public/{operator.bucket}/{file_name}"        
        doc_id = operator.upsert_document(file_name, body, content_hash, embedding_avg, pdf_url)

        # 8: Insert chunk and chunk_embeddings
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = operator.upsert_chunk(doc_id, idx, chunk["content"], chunk["page_number"])
            operator.insert_chunk_embedding(chunk_id, emb)
        
        tqdm.write(f"[DONE] Uploaded '{file_name}' with {len(chunks)} chunks")
        uploaded += 1

        # Clean temp dir
        os.remove(pdf_path)

    operator.conn.commit()

    # 9: check for deleted files
    print("\n[INFO] Checking for deleted files...")
    deleted = operator.sync_deletions()

    # 10: assign topics to chunks and propagate to documents
    print("[INFO] Initializing topics classifier...")
    init_topic_classifier_from_db()  # Train/Load Model
    
    print("[INFO] Assigning topics to chunks...")
    # assign_topics_to_chunks(overwrite=True)  # Classify all chunks
    assign_topics_to_chunks(overwrite=False) # False because we dont want to reassign already assigned topic to already ingested chunks
    
    # Final Summary
    print(f"\n{'='*60}")
    print(f"Ingestion Complete:")
    print(f"  ✅ New files uploaded: {uploaded}")
    print(f"  🔄 Files updated: {updated}")
    print(f"  ⏭️  Exact duplicates skipped: {skipped_identical}")
    print(f"  🗑️  Files deleted from DB: {deleted}")
    print(f"{'='*60}")

    # Clean temporary directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    ingest_documents()
