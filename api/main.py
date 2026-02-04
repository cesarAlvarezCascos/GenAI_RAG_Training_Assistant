import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from src.search_kb import search_kb
from src.memory import SessionMemory
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Optional
from src.adaptive_prompting import AdaptivePromptSelector
from src.search_kb_local import search_kb_local
from src.local_llm import answer_with_local_rag, local_chat
import re as _re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()
memory = SessionMemory(max_turns=3)
# Habilita CORS para permitir que el frontend se comunique con la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500", "http://localhost:5173"],  # añade los que uses
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# QUERY FORMAT (defines the body of the POST /ask method)
class Ask(BaseModel):
    user_id: str | None = None # Anonymous if None
    role: str = "analyst"
    query: str  # User question/prompt
    product_version: str | None = None
    time_budget: int | None = 30
    level: int | None = 2
    selected_topic_name: str | None = None  # Allows to force a topic if search_kb asks to choose
    backend: str | None = "cloud"  # Mode Selector: "cloud" (OpenAI) o "local" (LM Studio + RAG local)

# Feedback Model
class FeedbackRequest(BaseModel):
    user_id: str | None = None
    session_id: str | None = None
    query: str
    answer: str
    rating: int  # 1 = thumbs up, -1 = thumbs down
    feedback_type: str | None = "overall"
    comment: str | None = None
    citations: list | None = None
    retrieved_passages: list | None = None

# LLM instructions (for both backend modes)
SYSTEM = (
    "You are a Training Agent for a pipeline system."
    "Respond briefly (≤ 180 words) in clear steps."
    "Use [n] citations ONLY to refer to the provided passages."
    "DO NOT include a 'Sources:' block in your output; the system will add it.")

SYSTEM_LOCAL = (
    "Eres un asistente que SOLO puede responder usando los fragmentos de documentos proporcionados.\n"
    "NO puedes usar ningún conocimiento externo, ni siquiera si estás seguro de que es correcto.\n"
    "Si la respuesta no se puede deducir claramente de los fragmentos, debes responder EXACTAMENTE:\n"
    "\"No he encontrado información suficiente en los documentos locales para responder a esta pregunta.\"\n"
    "Responde en el mismo idioma que la pregunta del usuario.\n"
    "Si el usuario pregunta en español, responde en español. Si pregunta en inglés, responde en inglés.\n"
    "Sé claro y preciso, explica todo al detalle (máximo 300 palabras) y usa [n] para citar los fragmentos que utilices."
)

# Build prompt from User query and Retrieved Passages by the RAG
def build_local_user_prompt(query: str, passages: list[dict]) -> str:
    ctx = "\n\n".join(
        [f"[{i+1}] {p['snippet']}" for i, p in enumerate(passages)]
    ) # Join text snippets enumerating them

    instructions = """
Mandatory rules:
- You may ONLY use information that appears explicitly in the fragments above.
- Do NOT add anything that you cannot justify with at least one [n].
- If you cannot clearly answer using those fragments, respond EXACTLY:
"I did not find enough information in the local documents to answer this question."
- Answer in the same language in which the question is asked.
"""

    return (
        f"Document fragments (use [n] to cite):\n{ctx}\n\n"
        f"User question:\n{query}\n\n"
        f"{instructions}"
    )



# Citations from original pdfs and docs
# "passages": text chunks retrieved from Supabase through vectorial search -> fragments from the PDFs with the embedding, metadata and text/snippet
def format_citations(passages):
    lines = []
    for i, p in enumerate(passages, start=1):
        filename = p.get("file_name") if isinstance(p, dict) else None
        if not filename:
            name = "<unknown>"  # placeholder
        else:
            name = filename
        lines.append(f"[{i}] {name}")
    return "\n".join(lines)

# Selector de prompts adaptativo
adaptive_selector = AdaptivePromptSelector(max_examples=2)

# Main Endpoint
@app.post("/ask")
def ask(req: Ask):
    user_id = req.user_id or "anonymous"

    # 1. Obtener estadísticas para ajustar comportamiento
    stats = adaptive_selector.get_feedback_stats()
    
    # 2. Recuperar ejemplos similares exitosos (adaptive few-shot)
    positive_examples = adaptive_selector.retrieve_positive_examples(req.query)
    few_shot_context = adaptive_selector.build_few_shot_context(positive_examples)

    # 3. Retrieve memory history
    past_turns = memory.get(user_id)
    context_history = "\n".join(
        [f"Q: {t['query']}\nA: {t['answer']}" for t in past_turns]
    )

    backend = (req.backend or "cloud").lower()

    # ================== MODO LOCAL ==================
    if backend == "local":
        # 4L. Buscar en el RAG local
        passages = search_kb_local(req.query, top_k=6)
        if not passages:
            # Si el índice local no tiene nada relevante, no llamamos al modelo
            answer_text = "No he encontrado información suficiente en los documentos locales para responder a esta pregunta."
            final_answer = answer_text + "\n\nFuentes:\n" + format_citations([])
            return {
                "answer": final_answer,
                "citations": [],
                "adaptive_stats": stats,
            }

        # 5L. Llamar al modelo local con RAG estricto
        try:
            text, used_passages = answer_with_local_rag(req.query, passages)
        except Exception as e:
            text = f"Error generating local response: {type(e).__name__}."
            used_passages = passages  # por si acaso

        # 6L. Limpiar posibles bloques de 'Fuentes:' que el modelo invente
        text = _re.sub(
            r"\n+Fuentes:\s*(?:\[[^\]]+\].*|\S.*)+$",
            "",
            text,
            flags=_re.IGNORECASE | _re.DOTALL,
        )

        # 7L. Formatear respuesta final con citas
        final_answer = text + "\n\nFuentes:\n" + format_citations(used_passages[:6])

        # Guardar en memoria (solo texto)
        memory.add(user_id, req.query, text)

        return {
            "answer": final_answer,
            "citations": used_passages[:6],
            "adaptive_stats": stats,
        }
    
    
    # 4. Search knowledge base
    search_result = search_kb(req.query, selected_topic_name=req.selected_topic_name)
    if isinstance(search_result, dict) and search_result.get("status") == "choose_topic":
        return search_result # user selects topic

    # 5. Retrieve relevant knowledge base snippets
    passages = search_result
    if not passages:
        return {"answer": "I didn’t find any relevant matches in the database.", "citations": []}

    # 6. Build the augmented prompt + few-shot examples
    ctx = "\n\n".join([f"[{i+1}] {p['snippet']}" for i, p in enumerate(passages[:6])])

    # Ajustar instrucciones según satisfaction rate
    tone_adjustment = ""
    print(f"Satisfaction rate: {stats['satisfaction_rate']}")
    if stats['satisfaction_rate'] < 0.65 and stats['total'] > 10:
        tone_adjustment = "\n- IMPORTANT: Users have reported unsatifaction. Be more clear and specific."
    
    prompt = (
        f"{SYSTEM}\n\n"
        f"{few_shot_context}\n\n" # Few-shot examples
        f"Conversation history:\n{context_history or '(none)'}\n\n"
        f"User’s question: {req.query}\n\n"
        f"Relevant passages (use [n] to cite them in your answer):\n"
        f"{ctx}\n\n"
        "Instructions:\n"
        "- Summarise and answer in no more than 180 words.\n"
        "- Respond in the same language in which you are asked.\n"
        "- Insert [n] exactly where you use a fact from a passage.\n"
        "- DO NOT invent sources or add 'Sources:' at the end.\n"
        "- Include at least two citations if there are ≥ 2 passages.\n"
        f"{tone_adjustment}" # si la satisfacción es baja, ajustar tono (más claro y específico)
    )

    # Generate with OpenAI
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.choices[0].message.content.strip()
    except Exception as e:
        text = f"Error generating response: {type(e).__name__}."

    # Clean and finalize
    text = _re.sub(r"\n+Fuentes:\s*(?:\[[^\]]+\].*|\S.*)+$", "", text, flags=_re.IGNORECASE | _re.DOTALL)
    if text.count('[') < 2 and len(passages) >= 2:
        text += " [1][2]"

    final_answer = text + "\n\nFuentes:\n" + format_citations(passages[:6])

    # Save to memory
    memory.add(user_id, req.query, text)

    return {"answer": final_answer, "citations": passages[:6],
        "adaptive_stats": stats # para debug (opcional)
        }

# Añade este endpoint después de /ask
@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    """Captura feedback del usuario para mejorar el sistema"""
    try:
        from supabase import create_client
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )

        # Generar embedding de la consulta
        query_embedding = None
        try:
            embedding_response = client.embeddings.create(
                model="text-embedding-3-small",
                input=req.query
            )
            query_embedding = embedding_response.data[0].embedding
        except Exception as e:
            print(f"Warning: Could not generate embedding: {e}")

        # query_embedding = embedding_response.data[0].embedding
        data = {
            "user_id": req.user_id or "anonymous",
            "session_id": req.session_id,
            "query": req.query,
            "answer": req.answer,
            "rating": req.rating,
            "feedback_type": req.feedback_type,
            "comment": req.comment,
            "citations": req.citations,
            "retrieved_passages": req.retrieved_passages,
            "model_used": "gpt-4o-mini",
            "query_embedding": query_embedding # NUEVO: guardar embedding de la consulta
        }
        
        result = supabase.table("feedback").insert(data).execute()
        
        if not result.data:
            return {
                "status": "warning",
                "message": "Feedback not saved - verify RLS policies"
            }
        
        return {
            "status": "success",
            "message": "Feedback successfully saved",
            "feedback_id": result.data[0]["id"]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error al guardar feedback: {str(e)}"
        }


@app.get("/feedback/stats")
def get_feedback_stats():
    """Estadísticas incluyendo cobertura de embeddings"""
    try:
        from supabase import create_client
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        
        positive = supabase.table("feedback").select("*").eq("rating", 1).execute()
        negative = supabase.table("feedback").select("*").eq("rating", -1).execute()
        
        # Contar embeddings
        with_embeddings = supabase.table("feedback")\
            .select("id")\
            .not_.is_("query_embedding", "null")\
            .execute()
        
        total = len(positive.data) + len(negative.data)
        
        return {
            "total_feedback": total,
            "positive": len(positive.data),
            "negative": len(negative.data),
            "satisfaction_rate": len(positive.data) / total * 100 if total > 0 else 0,
            "embeddings_coverage": len(with_embeddings.data) / total * 100 if total > 0 else 0,
            "embeddings_count": len(with_embeddings.data)
        }
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/adaptive/status")
def adaptive_status():
    """Muestra estado del sistema adaptativo"""
    try:
        from supabase import create_client
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        stats = adaptive_selector.get_feedback_stats()
        
        # Obtener últimos 5 ejemplos positivos
        recent_positive = supabase.table("feedback")\
            .select("query")\
            .eq("rating", 1)\
            .order("created_at", desc=True)\
            .limit(5)\
            .execute()
        
        return {
            "feedback_stats": stats,
            "recent_positive_queries": [r['query'] for r in recent_positive.data],
            "adaptive_enabled": True,
            "max_examples": adaptive_selector.max_examples
        }
    except Exception as e:
        return {"error": str(e)}