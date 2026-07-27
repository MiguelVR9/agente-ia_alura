import os
from typing import Dict, Any, List
from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer

import config
from ingestion import load_all_documents, tokenize_technical_text

# ================================
# MODELO DE EMBEDDINGS (SINGLETON)
# ================================
_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    """Carga y reutiliza la instancia del modelo SentenceTransformer."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedding_model

# ================================
# CONSTRUCCIÓN DEL ÍNDICE BM25
# ================================
def build_bm25_index(chunks: List[str]) -> BM25Okapi:
    """Tokeniza el corpus usando el tokenizador técnico y genera el índice BM25."""
    tokenized_corpus = [tokenize_technical_text(doc) for doc in chunks]
    return BM25Okapi(tokenized_corpus)

# ================================
# GESTIÓN DE CHROMADB Y RECURSOS
# ================================
def initialize_indexes(force_reindex: bool = config.FORCE_REINDEX) -> Dict[str, Any]:
    """
    Orquesta la inicialización optimizada del RAG:
    - Si la colección existe y no se fuerza reindexación, recupera documentos directamente de ChromaDB.
    - Si se fuerza reindexación o no existe, ejecuta el pipeline de ingestión completo.
    - Retorna un diccionario estructurado de recursos.
    """
    print("Inicializando recursos de indexación...")
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
    model = get_embedding_model()

    if force_reindex:
        print(f"Reindexación forzada activa: Eliminando colección '{config.CHROMA_COLLECTION_NAME}'...")
        try:
            chroma_client.delete_collection(config.CHROMA_COLLECTION_NAME)
        except Exception:
            pass

    try:
        collection = chroma_client.get_collection(name=config.CHROMA_COLLECTION_NAME)
        print(f"✓ Colección existente detectada en ChromaDB ({collection.count()} registros).")
        
        # Recuperamos los documentos guardados para alimentar a BM25 sin releer los archivos
        data = collection.get(include=["documents", "metadatas"])
        chunks = data["documents"]
        metadatos = data["metadatas"]
        ids = data["ids"]

    except Exception:
        print("Colección no encontrada o reindexación requerida. Ejecutando pipeline de ingestión...")
        chunks, metadatos, ids = load_all_documents()

        collection = chroma_client.create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        passages = [f"passage: {doc}" for doc in chunks]
        embeddings = model.encode(
            passages, 
            batch_size=config.EMBEDDING_BATCH_SIZE, 
            show_progress_bar=True
        ).tolist()

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatos,
            ids=ids
        )
        print("✓ Indexación vectorial en ChromaDB completada con éxito.")

    print("Construyendo/Sincronizando índice lexical (BM25)...")
    bm25 = build_bm25_index(chunks)

    return {
        "bm25": bm25,
        "collection": collection,
        "embedding_model": model,
        "chunks": chunks,
        "metadatos": metadatos,
        "ids": ids
    }

if __name__ == "__main__":
    resources = initialize_indexes()
    print("\n===== RESUMEN DE RECURSOS =====")
    print(f"• Items en BM25: {len(resources['chunks'])}")
    print(f"• Registros en ChromaDB: {resources['collection'].count()}")
    print("✓ Módulo indexer.py perfeccionado.")
