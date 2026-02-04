import supabase
import os
from dotenv import load_dotenv
import time
import psycopg
import hashlib
import shutil
import tqdm
from pathlib import Path
import glob

class DatabaseOperator():
    def __init__(self, url: str, key: str, path: str | None = None):
        load_dotenv()
        self.dsn = os.getenv("DATABASE_URL")  # ----- nuevo
        self.conn = psycopg.connect(self.dsn)
        self.client = supabase.create_client(url, key)
        self.bucket = "pdfs"

        # self.data_dir = path
        self.pdf_files = self._get_pdf_files()

    # -------- nuevo
    def ensure_conn(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
        except psycopg.OperationalError:
            self.conn = psycopg.connect(self.dsn)
    
    
    def download_from_supabase(self, file_name, temp_dir="./temp_pdfs"):  
        '''
        Download from Supabase Storage to temp local Dir.
        As hashes, chunks and embeddings are computed in our machine, it needs
        the files locally during that process
        '''
    
        # Create temp dir if not existing
        os.makedirs(temp_dir, exist_ok=True)
        local_path = os.path.join(temp_dir, file_name)

        try:
            # Download file
            res = self.client.storage.from_(self.bucket).download(file_name)

            # Save temporarly
            with open(local_path, 'wb') as f:
                f.write(res)

            return local_path

        except Exception as e:
            print(f"[ERROR] Failed to download {file_name}: {e}")
            return None
        

    def compute_pdf_hash(self, file_path: str) -> str:
        """Compute hash SHA256 of PDF binary content"""

        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    

    def hash_exists(self, content_hash):
        """Check Existing / Duplicates
        Verifies if an existent document already has this hash """

        self.ensure_conn()  # --- nuevo
        with self.conn.cursor() as cur:
            cur.execute("SELECT id, file_name FROM documents WHERE content_hash = %s", (content_hash,))
            return cur.fetchone()
    

    def archive_old_version(self, file_name: str) -> str:
        """Move old PDF (when it has been replaced by an updated version) to folder /pdfs/archived/ in Supabase Storage bucket"""

        try:
            # 1. Download actual file (in memory)
            res = self.client.storage.from_(self.bucket).download(file_name)

            # 2. Upload actual file to archived/ 
            archived_path = f"archived/{file_name}"
            self.client.storage.from_(self.bucket).upload(
                path = archived_path,
                file = res,
                file_options = {"content-type": "application/pdf", "upsert": "true" }
                )
            
            # 3. Delete original file (from root pdfs/)
            self.client.storage.from_(self.bucket).remove([file_name])

            print(f"[ARCHIVED] Moved {file_name} to archived/ in bucket Storage")
            return archived_path
        
        except Exception as e:
            print(f"[ERROR] Failed to archive {file_name} in Storage: {e}")
            return None
    

    def find_similar_document(self, embedding_avg, file_name: str, threshold=0.95):
        """
        Detects an updated version using:
        Doc. Embedding Similarity: looks for documents with similar avg embedding (> threshold)
        Filename Similarity: using pg_trm with 0.5 similarity in the filename
        Returns (doc_id, file_name, similarity) of the most similar.
        """
        self.ensure_conn() # ----- nuevo
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    id, 
                    file_name,
                    1 - (embedding_avg <=> %s::vector) AS sim_embedding,
                    similarity(file_name, %s) AS sim_filename
                FROM documents
                WHERE embedding_avg IS NOT NULL
                ORDER BY sim_embedding DESC
                LIMIT 1
            """, (embedding_avg, file_name))

            for doc_id, old_name, sim_emb, sim_fname in cur.fetchall():
                if sim_emb >= threshold and sim_fname > 0.5:
                    return (doc_id, old_name, sim_emb)  # (id, file_name, similarity)
            
            return None
        
    
    def delete_document_cascade(self, doc_id, file_name=None, should_archive = False):
        """ Deletes document and all its associated chunks/embeddings"""
        
        self.ensure_conn()  # --- nuevo
        with self.conn.cursor() as cur:
            # 1st delete chunk_embeddings
            cur.execute("""
                DELETE FROM chunk_embeddings 
                WHERE chunk_id IN (
                    SELECT id FROM chunks WHERE document_id = %s
                )
            """, (doc_id,))
            
            # Then delete chunks
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
            
            # Finally delete the document
            cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        self.conn.commit()

        # Delete from /pdfs 
        if file_name and should_archive:
            self.archive_old_version(file_name)

    
    def upsert_document(self, file_name, body, content_hash, embedding_avg, pdf_url):
        '''Insert doc in table 'documents' and return doc ID'''

        self.ensure_conn()  # --- nuevo
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO documents (file_name, body, content_hash, embedding_avg, pdf_url)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (file_name, body, content_hash, embedding_avg, pdf_url))
            doc_id = cur.fetchone()[0]
        self.conn.commit()
        return doc_id


    def upsert_chunk(self, document_id, idx, content, page_number):
        '''Insert chunk in table 'chunks' and return chink ID'''

        self.ensure_conn()  # --- nuevo
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chunks (document_id, chunk_idx, body, page_number)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (document_id, idx, content, page_number))
            chunk_id = cur.fetchone()[0]
        self.conn.commit()
        return chunk_id


    def insert_chunk_embedding(self, chunk_id, embedding):
        '''Insert a chunk embedding in table 'chunk_embeddings'''

        self.ensure_conn()  # --- nuevo
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chunk_embeddings (chunk_id, embedding)
                VALUES (%s, %s)
            """, (chunk_id, embedding))
        self.conn.commit()

    
    def sync_deletions(self):
        """
        Delete from Database documents that have been removed from /pdfs
        So if a PDF is manually deleted from the folder, it won't be used to generate answers
        """

        # List files in Storage pdfs/ 
        try:
            storage_files = self.client.storage.from_(self.bucket).list()
            # Only in root pdfs/ (not in pdfs/archived/)
            existing_names = {
                f['name'] for f in storage_files
                if f['name'].endswith('.pdf') and not f['name'].startswith('archived/')
            }
        except Exception as e:
            print(f"[ERROR] Could not list Storage files: {e}")
            return 0

        with self.conn.cursor() as cur:
            cur.execute("SELECT id, file_name FROM documents")
            db_files = cur.fetchall()
                
        deleted_count = 0
        for doc_id, file_name in db_files:
            if file_name not in existing_names:
                tqdm.write(f"[SYNC DELETE] Removing '{file_name}' from database (file no longer exists)")
                self.delete_document_cascade(doc_id)
                deleted_count += 1
        
        return deleted_count
    
    
    def _get_pdf_files(self):
        # List files in bucket Storage
        try:
            storage_files = self.client.storage.from_(self.bucket).list()
            # Filter only those in root /pdfs (not in archived/)
            pdf_files = [
                f['name'] for f in storage_files
                if f['name'].endswith('.pdf') and not f['name'].startswith('archived/')
            ]

            return pdf_files

        except Exception as e:
            print(f"[ERROR] Could not list Storage: {e}")
            return []
    