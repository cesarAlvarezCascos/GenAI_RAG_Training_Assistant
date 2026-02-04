# src/analyze_feedback.py
import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI
from collections import Counter
import json

# TRES MODOS---------------------------------------
# 1. Resumen rápido: python analyze_feedback.py summary
# 2. Clasificar sin actualizar BD: python analyze_feedback.py classify-only
# 3. Clasificar y actualizar BD: python analyze_feedback.py
# --------------------------------------------------

project_root = Path(__file__).parent.parent
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path)

# Inicializar clientes
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

def classify_feedback_issue(comment: str, query: str, answer: str) -> str:
    """
    Clasifica el problema del feedback usando GPT-4 en una de estas categorías:
    - relevance: La respuesta no es relevante a la pregunta
    - accuracy: La respuesta contiene información incorrecta o desactualizada
    - completeness: La respuesta está incompleta o le falta información importante
    """
    
    prompt = f"""Analiza este feedback negativo de un sistema RAG y clasifica el problema en UNA de estas tres categorías:

**Categorías:**
1. "relevance" - La respuesta no es relevante para la pregunta, se fue por las ramas, o responde algo diferente
2. "accuracy" - La respuesta contiene información incorrecta, desactualizada, o contradictoria
3. "completeness" - La respuesta es correcta pero incompleta, superficial, o le falta información importante

**Contexto:**
Pregunta del usuario: "{query}"
Respuesta del sistema: "{answer[:300]}..."
Comentario del usuario: "{comment}"

**Instrucciones:**
Analiza el comentario del usuario y determina cuál es el problema principal.
Responde SOLO con una de estas tres palabras: relevance, accuracy, o completeness
Ten en cuenta que este ejercicio es para el desarrollo de un RAG las respuestas deben 
pertenecer al contexto interno de la empresa SANDOZ y sus productos y usar las fuentes oficiales.

Clasificación:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de feedback de sistemas RAG. Clasificas problemas de manera precisa y concisa."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  # Determinístico
            max_tokens=10
        )
        
        classification = response.choices[0].message.content.strip().lower()
        
        # Validar que sea una de las tres categorías
        if classification not in ['relevance', 'accuracy', 'completeness']:
            print(f"⚠️ Invalid Classification '{classification}', using 'relevance' by default")
            return 'relevance'
        
        return classification
    
    except Exception as e:
        print(f"❌ Failed to classify: {e}")
        return 'relevance'  # Valor por defecto


def update_feedback_type(feedback_id: int, new_type: str) -> bool:
    """Actualiza el feedback_type en la base de datos"""
    try:
        result = supabase.table("feedback")\
            .update({"feedback_type": new_type})\
            .eq("id", feedback_id)\
            .execute()
        return True
    except Exception as e:
        print(f"❌ Failed to update feedback {feedback_id}: {e}")
        return False


def analyze_and_classify_feedback(auto_update: bool = True):
    """
    Analiza feedbacks negativos, clasifica los que tienen comentarios,
    y opcionalmente actualiza la base de datos
    """
    
    print("\n" + "="*70)
    print("📊 FEEDBACK ANALYSIS & CLASSIFICATION")
    print("="*70)
    
    # Obtener feedbacks negativos con comentarios
    negative_with_comments = supabase.table("feedback")\
        .select("*")\
        .eq("rating", -1)\
        .not_.is_("comment", "null")\
        .order("created_at", desc=True)\
        .execute()
    
    print(f"\n✅ There are {len(negative_with_comments.data)} negative feedbacks with comments\n")
    
    if len(negative_with_comments.data) == 0:
        print("No comments to analyze.")
        return
    
    # Clasificar cada comentario
    classifications = {
        'relevance': [],
        'accuracy': [],
        'completeness': []
    }
    
    for i, feedback in enumerate(negative_with_comments.data, 1):
        print(f"[{i}/{len(negative_with_comments.data)}] Analizando feedback ID: {feedback['id']}")
        print(f"   Query: {feedback['query'][:60]}...")
        print(f"   Comment: {feedback['comment'][:80]}...")
        
        # Clasificar el problema
        category = classify_feedback_issue(
            comment=feedback['comment'],
            query=feedback['query'],
            answer=feedback['answer']
        )
        
        print(f"   🏷️  Classification: {category}\n")
        
        classifications[category].append({
            'id': feedback['id'],
            'query': feedback['query'],
            'comment': feedback['comment']
        })
        
        # Actualizar en la base de datos si está habilitado
        if auto_update:
            success = update_feedback_type(feedback['id'], category)
            if success:
                print(f"   ✅ Database Updated\n")
    
    # Resumen de clasificaciones
    print("\n" + "="*70)
    print("📈 CLASSIFICATIONS SUMMARY")
    print("="*70 + "\n")
    
    for category, items in classifications.items():
        print(f"🔹 {category.upper()}: {len(items)} casos")
        for item in items[:3]:  # Mostrar solo los primeros 3
            print(f"   - {item['query'][:60]}...")
            print(f"     💬 {item['comment'][:80]}...\n")
        if len(items) > 3:
            print(f"   ... y {len(items) - 3} más\n")
    
    # Estadísticas finales
    total = sum(len(items) for items in classifications.values())
    print("\n" + "="*70)
    print("📊 STATISTICS")
    print("="*70)
    print(f"Total analyzed: {total}")
    print(f"Relevance: {len(classifications['relevance'])} ({len(classifications['relevance'])/total*100:.1f}%)")
    print(f"Accuracy: {len(classifications['accuracy'])} ({len(classifications['accuracy'])/total*100:.1f}%)")
    print(f"Completeness: {len(classifications['completeness'])} ({len(classifications['completeness'])/total*100:.1f}%)")
    print("="*70 + "\n")


def analyze_negative_feedback_summary():
    """Análisis rápido de todos los feedbacks negativos (con y sin comentarios)"""
    
    negative = supabase.table("feedback")\
        .select("*")\
        .eq("rating", -1)\
        .order("created_at", desc=True)\
        .execute()
    
    print(f"\n=== Resumen General: {len(negative.data)} feedbacks negativos ===\n")
    
    # Queries problemáticas
    queries = [f["query"] for f in negative.data]
    print("📌 Queries con más problemas:")
    for query, count in Counter(queries).most_common(5):
        print(f"  ({count}x) {query[:70]}...")
    
    # Distribución por tipo
    types = [f["feedback_type"] for f in negative.data if f.get("feedback_type")]
    if types:
        print(f"\n🏷️  Distribución por tipo:")
        for ftype, count in Counter(types).most_common():
            print(f"  {ftype}: {count}")
    
    # Comentarios
    comments = [f for f in negative.data if f.get("comment")]
    print(f"\n💬 Feedbacks con comentarios: {len(comments)}/{len(negative.data)}")

"""
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        # Modo resumen rápido
        analyze_negative_feedback_summary()
    elif len(sys.argv) > 1 and sys.argv[1] == "classify-only":
        # Solo clasificar, no actualizar BD
        analyze_and_classify_feedback(auto_update=False)
    else:
        # Modo completo: clasificar y actualizar
        analyze_and_classify_feedback(auto_update=True)"""
