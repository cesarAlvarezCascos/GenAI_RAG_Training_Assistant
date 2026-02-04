import os
from supabase import create_client
from openai import OpenAI
from typing import List, Dict

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

class AdaptivePromptSelector:
    """Selecciona ejemplos dinámicamente usando búsqueda vectorial optimizada"""
    
    def __init__(self, max_examples: int = 3, similarity_threshold: float = 0.7):
        self.max_examples = max_examples
        self.similarity_threshold = similarity_threshold
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Genera embedding de la consulta"""
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",  # 1536 dimensiones
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def retrieve_positive_examples(self, query: str) -> List[Dict]:
        """
        Recupera ejemplos similares usando búsqueda vectorial optimizada.
        Usa la función RPC de Supabase para aprovechar el índice IVF.

        La función RPC es una funcion PostgreSQL creada en Supabase con endpoint
        """
        try:
            # Generar embedding de la consulta
            query_embedding = self.get_query_embedding(query)
            if not query_embedding:
                return []
            
            # Llamar a la función RPC optimizada
            # Buscamos nº ejemplos de feedback positivo cuya query fue similar a la actual
            result = supabase.rpc(
                'match_positive_feedback',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': self.similarity_threshold,
                    'match_count': self.max_examples
                }
            ).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            print(f"Error retrieving examples: {e}")
            return []
    
    def build_few_shot_context(self, examples: List[Dict]) -> str:
        """Construye contexto few-shot desde ejemplos positivos
        
            Se construye el contexto few-shot, y se añade al prompt antes de lanzar la respuesta para la query actual
        """
        if not examples:
            return ""
        
        context_parts = ["Ejemplos de respuestas exitosas:\n"]
        for i, ex in enumerate(examples, 1):
            similarity_pct = ex.get('similarity', 0) * 100
            context_parts.append(
                f"\nEjemplo {i} (similitud: {similarity_pct:.1f}%):\n"
                f"Pregunta: {ex['query']}\n"
                f"Respuesta: {ex['answer'][:200]}...\n"
            )
        
        return "\n".join(context_parts)
    
    def get_feedback_stats(self) -> Dict:
        """Obtiene estadísticas de feedback guardado"""
        try:
            positive = supabase.table("feedback").select("id").eq("rating", 1).execute()
            negative = supabase.table("feedback").select("id").eq("rating", -1).execute()
            
            # Contar cuántos tienen embeddings
            with_embeddings = supabase.table("feedback")\
                .select("id")\
                .not_.is_("query_embedding", "null")\
                .execute()
            
            total = len(positive.data) + len(negative.data)
            return {
                "satisfaction_rate": len(positive.data) / total if total > 0 else 0,
                "total": total,
                "positive": len(positive.data),
                "negative": len(negative.data),
                "with_embeddings": len(with_embeddings.data)
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"satisfaction_rate": 0, "total": 0}
