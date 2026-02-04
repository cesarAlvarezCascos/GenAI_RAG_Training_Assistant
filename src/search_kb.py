import os, psycopg
from dotenv import load_dotenv
from openai import OpenAI
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
conn = psycopg.connect(os.getenv("DATABASE_URL"))


def _embed(q: str):
    """Create a 1536-dim embedding vector from the user query."""
    return client.embeddings.create(model="text-embedding-3-small", input=q).data[0].embedding


def vector_search(qvec, k=12, topic_id=None):
    """
    Perform similarity search using cosine distance on chunk embeddings.
    Joins `chunk_embeddings` -> `chunks` -> `documents`.

    If `topic_id` is provided it must be the topic's id (UUID/int), not
    the topic name. Passing a topic name here will cause a type error
    when compared against `c.topic_id` in Postgres.
    """
    with conn.cursor() as cur:
        if topic_id is not None:
            cur.execute("""
                SELECT 
                    c.id AS chunk_id,
                    c.body AS chunk_text,
                    d.id AS document_id,
                    d.file_name,
                    c.page_number,
                    d.pdf_url,
                    1 - (ce.embedding <=> %s::vector) AS vscore
                FROM chunk_embeddings ce
                JOIN chunks c ON ce.chunk_id = c.id
                JOIN documents d ON c.document_id = d.id
                WHERE c.topic_id = %s
                ORDER BY ce.embedding <=> %s::vector
                LIMIT %s;
            """, (qvec, topic_id, qvec, k))
        else:
            cur.execute("""
                SELECT 
                    c.id AS chunk_id,
                    c.body AS chunk_text,
                    d.id AS document_id,
                    d.file_name,
                    c.page_number,
                    d.pdf_url,
                    1 - (ce.embedding <=> %s::vector) AS vscore
                FROM chunk_embeddings ce
                JOIN chunks c ON ce.chunk_id = c.id
                JOIN documents d ON c.document_id = d.id
                ORDER BY ce.embedding <=> %s::vector
                LIMIT %s;
            """, (qvec, qvec, k))
        return cur.fetchall()



def keyword_search(query, k=12, topic_id=None):
    """
    Perform keyword-based search using pg_trgm similarity between query text 
    and chunks. If a topic_id is provided, restrict search to that topic.
    """
    with conn.cursor() as cur:
        if topic_id:
            cur.execute("""
                SELECT 
                    c.id AS chunk_id,
                    c.body AS chunk_text,
                    d.id AS document_id,
                    d.file_name,
                    c.page_number,
                    d.pdf_url,
                    similarity(c.body, %s) AS kscore
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.topic_id = %s
                AND similarity(c.body, %s) > 0.1
                ORDER BY kscore DESC
                LIMIT %s;
            """, (query, topic_id, query, k))
        else:
            cur.execute("""
                SELECT 
                    c.id AS chunk_id,
                    c.body AS chunk_text,
                    d.id AS document_id,
                    d.file_name,
                    c.page_number,
                    d.pdf_url,
                    similarity(c.body, %s) AS kscore
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE similarity(c.body, %s) > 0.1
                ORDER BY kscore DESC
                LIMIT %s;
            """, (query, query, k))
        return cur.fetchall()
    


def fuse(vec_hits, key_hits, alpha=0.6, top_k=8, relevance_threshold=0.1):
    """
    Combine vector search and keyword search scores using a weighted fusion.
    alpha controls the weight given to vector similarity.

    Returns the top_k Chunks info (Citations data)
    """
    S = defaultdict(float) # Weighted sum of scores per chunk
    M = {}  # chunk snippet (txt), source doc (filename), doc id (did) and page number (page_num) for each chunk

    for (cid, txt, did, filename, page_num, pdf_url, sc) in vec_hits:
        if cid not in M:
            M[cid] = (txt, filename, did, page_num, pdf_url)
        S[cid] += alpha * float(sc)

    for (cid, txt, did, filename, page_num, pdf_url, sc) in key_hits:
        if cid not in M:
            M[cid] = (txt, filename, did, page_num, pdf_url)
        S[cid] += (1 - alpha) * float(sc)

    # Apply relevance threshold
    S = {cid: score for cid, score in S.items() if score >= relevance_threshold}
    if len(S) == 0:
        return []

    ranked = sorted(S.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Return the info we want in the Citations JSON
    return [
        {
            # Ensure values are JSON-serializable (UUIDs -> str)
            "chunk_id": str(cid),
            "file_name": str(M[cid][1]),
            "document_id": str(M[cid][2]),
            "page_number": (M[cid][3] + 1) if M[cid][3] is not None else None,
            "pdf_url": M[cid][4],
            "snippet": M[cid][0][:600],
            "score": float(s)
        }
        for cid, s in ranked
    ]


def detect_topic(query: str):
    '''
    used to focus only on the chunks of the detected topic depending on the query, so only
    chunks from that topic will be retrieved
    '''

    with conn.cursor() as cur:
        cur.execute("SELECT id, name, description, key_words FROM topics;")
        topics = cur.fetchall()  # All topics; each row: (id, name, description, [keywords])

    query_emb = _embed(query)  # Query Embedding
    scored_topics = []
    descriptions = {}  # NUEVO: guardar descripción por topic_id

    for tid, name, description, keywords in topics:
        # representative text for each topic
        topic_text = f"Name: {name}\nDescription: {description}\nKeywords: {', '.join(keywords or [])}"
        topic_emb = _embed(topic_text)

        # Cosine similarity between query embedding and topic_text embedding 
        sim = dot(query_emb, topic_emb) / (norm(query_emb) * norm(topic_emb) + 1e-8)

        #final_score = 0.8 * sim + 0.2 * kw_score_norm
        final_score = sim
        scored_topics.append((tid, name, final_score))

        # guardar descripción
        descriptions[tid] = description or ""

    # Ranking of topics for the Query
    scored_topics.sort(key=lambda x: x[2], reverse=True)

    if not scored_topics:
        return None, []

    # Top 1 Topic (most similar to the query)
    best_tid, best_name, best_score = scored_topics[0]
    second_score = scored_topics[1][2] if len(scored_topics) > 1 else 0

    # if the best score is not higher than 0.55 or the difference between the 2 first scores is small e.g. 0.1 
    if best_score < 0.55 or (best_score - second_score < 0.10):
        all_topics = [
            {
                "name": name,
                "description": descriptions.get(tid, "")
            }
            for tid, name, _ in scored_topics
        ]
        return None, all_topics

    return (best_tid, best_name), []


def search_kb(query: str, top_k: int = 8, selected_topic_name: str | None = None):
    """
    0. Detect topic (if any)
    1. Embed query → vector search
    2. Keyword search → fuse both
    """
    if selected_topic_name:
        # the user selects manually a topic because detect_topic fails
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM topics WHERE name = %s", (selected_topic_name,))
            row = cur.fetchone()
            topic_id = row[0] if row else None
            topic_name = selected_topic_name if row else None
    else:
        # try to detect topic using detect_topic
        matched_topic, all_topics = detect_topic(query)
        if matched_topic is None:
            # if no topic selected → frontend must ask
            return {
                "status": "choose_topic",
                "message": "A topic has not been detected. Please, select one of the following topics.",
                "topics": all_topics
            } 
        topic_id, topic_name = matched_topic # This is the Top-1 similar topic

    qvec = _embed(query)
    vec_hits = vector_search(qvec, k=top_k * 2, topic_id=topic_id)
    key_hits = keyword_search(query, k=top_k * 2, topic_id=topic_id)
    results = fuse(vec_hits, key_hits, alpha=0.6, top_k=top_k)
    return results
