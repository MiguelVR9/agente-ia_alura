import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import CrossEncoder

import config
from ingestion import tokenize_technical_text, expandir_query

# ================================
# MODELO CROSS-ENCODER (SINGLETON)
# ================================
_reranker_model = None

def get_reranker_model() -> CrossEncoder:
    """Carga y reutiliza la instancia del modelo Reranker."""
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(config.RERANKER_MODEL_NAME)
    return _reranker_model

# ================================
# VALIDACIÓN DE CONFIANZA
# ================================
def es_respuesta_valida(candidatos: List[Dict[str, Any]], score_top1: float) -> bool:
    """Valida si el candidato top-1 supera el umbral de confianza calibrado."""
    if not candidatos:
        return False
    if score_top1 < config.THRESHOLD_CONFIANZA:
        return False
    return True

# ================================
# BÚSQUEDA LÉXICA (BM25)
# ================================
def search_bm25(
    query: str, 
    bm25_index: Any, 
    chunks: List[str], 
    metadatos: List[Dict[str, Any]], 
    ids: List[str], 
    k: int = config.TOP_K_BM25
) -> List[Dict[str, Any]]:
    """Ejecuta búsqueda lexical con BM25 tras expandir y tokenizar la query."""
    query_expandida = expandir_query(query)
    tokenized_query = tokenize_technical_text(query_expandida)
    scores = bm25_index.get_scores(tokenized_query)
    top_k_indices = np.argsort(scores)[::-1][:k]
    
    results = []
    for idx in top_k_indices:
        results.append({
            "id": ids[idx],
            "document": chunks[idx],
            "metadata": metadatos[idx],
            "bm25_score": float(scores[idx])
        })
    return results

# ================================
# BÚSQUEDA VECTORIAL (CHROMADB)
# ================================
def search_vectorial(
    query: str, 
    collection: Any, 
    embedding_model: Any, 
    k: int = config.TOP_K_VECTOR
) -> List[Dict[str, Any]]:
    """Ejecuta búsqueda semántica en ChromaDB anteponiendo 'query: ' al texto."""
    query_text = f"query: {query}"
    query_embedding = embedding_model.encode(query_text).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    vector_results = []
    if results and results.get("ids") and len(results["ids"]) > 0:
        for i in range(len(results["ids"][0])):
            vector_results.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "vector_distance": results["distances"][0][i]
            })
    return vector_results

# ================================
# RECIPROCAL RANK FUSION (RRF)
# ================================
def reciprocal_rank_fusion(
    bm25_results: List[Dict[str, Any]], 
    vector_results: List[Dict[str, Any]], 
    k_rrf: int = config.RRF_K, 
    top_n: int = config.RRF_TOP_N
) -> List[Dict[str, Any]]:
    """Combina y reordena los resultados de BM25 y vectorial usando RRF."""
    rrf_scores = {}
    item_map = {}

    for rank, item in enumerate(bm25_results):
        doc_id = item["id"]
        item_map[doc_id] = item
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rrf + rank + 1))

    for rank, item in enumerate(vector_results):
        doc_id = item["id"]
        if doc_id not in item_map:
            item_map[doc_id] = item
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rrf + rank + 1))

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    fused_list = []
    for doc_id, rrf_score in sorted_docs:
        base_item = item_map[doc_id]
        base_item["rrf_score"] = rrf_score
        fused_list.append(base_item)
        
    return fused_list

# ================================
# ORQUESTADOR DE RETRIEVAL
# ================================
def ejecutar_retrieval_rag(
    query: str, 
    rag_resources: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], float, bool]:
    """
    Ejecuta el pipeline completo de Búsqueda Híbrida + RRF + CrossEncoder Reranker.
    Acepta el diccionario de recursos entregado por indexer.py.
    """
    bm25_index = rag_resources["bm25"]
    collection = rag_resources["collection"]
    embedding_model = rag_resources["embedding_model"]
    chunks = rag_resources["chunks"]
    metadatos = rag_resources["metadatos"]
    ids = rag_resources["ids"]

    # 1. Búsquedas individuales
    res_bm25 = search_bm25(query, bm25_index, chunks, metadatos, ids, k=config.TOP_K_BM25)
    res_vec = search_vectorial(query, collection, embedding_model, k=config.TOP_K_VECTOR)
    
    # 2. Fusión RRF
    candidatos = reciprocal_rank_fusion(res_bm25, res_vec, k_rrf=config.RRF_K, top_n=config.RRF_TOP_N)
    
    if not candidatos:
        return [], 0.0, False

    # 3. Reranking con CrossEncoder evaluando contra 'texto_raw'
    reranker_model = get_reranker_model()
    pairs = [[query, doc["metadata"]["texto_raw"]] for doc in candidatos]
    raw_scores = reranker_model.predict(pairs)
    
    # Normalización sigmoide para convertir a puntaje de confianza [0, 1]
    scores_norm = 1 / (1 + np.exp(-np.array(raw_scores)))

    for i, doc in enumerate(candidatos):
        doc["cross_encoder_raw"] = float(raw_scores[i])
        doc["confidence_score"] = float(scores_norm[i])

    # 4. Ordenamiento final y selección de Top K
    candidatos_ordenados = sorted(candidatos, key=lambda x: x["confidence_score"], reverse=True)[:config.TOP_K_FINAL]
    top_score = candidatos_ordenados[0]["confidence_score"] if candidatos_ordenados else 0.0
    
    confiable = es_respuesta_valida(candidatos_ordenados, top_score)
    return candidatos_ordenados, top_score, confiable

# ================================
# CONSTRUCCIÓN DE PROMPT LLM
# ================================
def construir_prompt_para_llm(
    pregunta: str, 
    documentos_recuperados: List[Dict[str, Any]], 
    es_confiable: bool
) -> Tuple[Optional[str], Optional[str]]:
    """Construye el prompt formal para la API del LLM o retorna mensaje de rechazo."""
    if not es_confiable:
        return None, "No se encontró información técnica suficiente en los manuales ni fichas técnicas para responder a esta consulta con la precisión requerida."
    
    bloques_contexto = []
    for d in documentos_recuperados:
        meta = d["metadata"]
        ref = f"{meta['tipo_documento']} (Página/Código: {meta.get('pagina', meta.get('codigo'))}, ID: {meta['chunk_id']})"
        bloques_contexto.append(f"--- FUENTE: {ref} ---\n{meta['texto_raw']}")
        
    contexto_str = "\n\n".join(bloques_contexto)
    
    prompt = f"""
Eres un asistente técnico especializado en recubrimientos industriales RIS.
Tu tarea es responder la consulta del usuario ÚNICAMENTE utilizando los fragmentos de contexto técnico proporcionados a continuación.

REGLAS DE GENERACIÓN:
1. Responde de forma técnica, precisa y directa.
2. Para cada afirmación importante, indica explícitamente la fuente utilizada (Ejemplo: [Manual Operativo, Página 6]).
3. Si la información solicitada no está explícitamente en el contexto, indica estrictamente: "No dispongo de información suficiente en la documentación para responder a este punto."

CONTEXTO TÉCNICO:
{contexto_str}

PREGUNTA DEL USUARIO:
{pregunta}

RESPUESTA TÉCNICA:
"""
    return prompt, None

if __name__ == "__main__":
    from indexer import initialize_indexes
    print("Probando retrieval.py de forma autónoma...")
    resources = initialize_indexes()
    q = "¿Cuál es el procedimiento y preparación para corrosión severa con chorro de arena?"
    docs, score, confiable = ejecutar_retrieval_rag(q, resources)
    print(f"✓ Consulta de prueba ejecutada con éxito. Top-1 score: {score:.4f} | Confiable: {confiable}")
