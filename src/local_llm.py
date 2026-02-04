# src/local_llm.py
import os
import requests
from typing import Dict, List, Tuple

# URL del servidor de LM Studio (OpenAI compatible)
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen2.5-1.5B-Instruct-MLX-4bit")

# Si has puesto API key en LM Studio, ponla en esta env var
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY")


def local_chat(system_prompt: str, user_content: str) -> str:
    """
    Llama al modelo local expuesto por LM Studio
    usando /v1/chat/completions (OpenAI compatible).
    NO hace ninguna lógica de RAG, solo envía prompts.
    """
    url = f"{LMSTUDIO_BASE_URL}/v1/chat/completions"

    headers: Dict[str, str] = {
        "Content-Type": "application/json",
    }
    if LMSTUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {LMSTUDIO_API_KEY}"

    payload: Dict = {
        "model": LOCAL_LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    return data["choices"][0]["message"]["content"].strip()


def answer_with_local_rag(query: str, passages: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Construye un prompt RAG estricto con los pasajes recuperados
    y devuelve la respuesta del modelo + los propios pasajes.

    Espera que cada passage tenga al menos:
      - 'snippet'
      - 'file_name'
      - 'page_number'
    """
    # Si no hay pasajes, mensaje estándar
    if not passages:
        msg = "No he encontrado información suficiente en los documentos locales para responder a esta pregunta."
        return msg, passages

    # Construimos el contexto numerado [1], [2], ...
    ctx_blocks = []
    for i, p in enumerate(passages):
        snippet = p.get("snippet", "")
        ctx_blocks.append(f"[{i+1}] {snippet}")
    ctx = "\n\n".join(ctx_blocks)

    system_prompt = (
        "Eres un asistente RAG ESTRICTO para documentos internos de Sandoz. "
        "Solo puedes usar información que aparezca en los fragmentos proporcionados. "
        "Si los fragmentos no contienen información suficiente para responder, "
        "responde EXACTAMENTE: "
        "'No he encontrado información suficiente en los documentos locales para responder a esta pregunta.' "
        "Si sí hay información relevante, responde de forma breve (máx. 180 palabras), clara y estructurada, "
        "citando los fragmentos usados con [1], [2], etc."
    )

    user_content = (
        f"Pregunta del usuario:\n{query}\n\n"
        f"Fragmentos relevantes:\n{ctx}\n\n"
        "Responde usando EXCLUSIVAMENTE la información de estos fragmentos. "
        "Si no alcanzan para responder, usa la frase exacta indicada en el sistema."
    )

    answer = local_chat(system_prompt, user_content)
    return answer, passages
