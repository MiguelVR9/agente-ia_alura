import os
from pathlib import Path

# ================================
# PROYECTO Y RUTAS BASE
# ================================
PROJECT_NAME = "RIS_RAG_Assistant"
VERSION = "1.0.0"

# Obtiene la raíz del proyecto (mi-proyecto-rag/) desde src/config.py
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = str(BASE_DIR / "chroma_db_ris")
CHROMA_COLLECTION_NAME = "ris_knowledge_base"

LOG_DIR = BASE_DIR / "logs"
RUTA_LOGS = str(LOG_DIR / "historial_evaluacion_rag.csv")
FEEDBACK_PATH = str(LOG_DIR / "feedback_usuarios.csv")

# Creación automática de directorios para evitar errores en ejecuciones limpias
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ================================
# DOCUMENTOS FUENTE
# ================================
FICHAS_TECNICAS_PATH = str(DATA_DIR / "Fichas_Tecnicas_Materiales_RIS.pdf")
MANUALES_PATH = str(DATA_DIR / "Manuales_Procesos_Procedimientos_RIS.pdf")
MATERIALES_EXCEL_PATH = str(DATA_DIR / "Especificaciones_Materiales_RIS.xlsx")
EQUIPOS_EXCEL_PATH = str(DATA_DIR / "Especificaciones_Equipos_RIS.xlsx")

# ================================
# MODELOS IA Y LLM
# ================================
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

LLM_MODEL_NAME = "gemini-3.5-flash"
LLM_TEMPERATURE = 0.0
LLM_MAX_RETRIES = 2
MAX_REINTENTOS_API = 2

# ================================
# CHUNKING
# ================================
CHUNK_SIZE = 650
CHUNK_OVERLAP = 120

IGNORE_TOP_PAGES_MANUAL = 3
IGNORE_TOP_PAGES_FICHAS = 1

# ================================
# RETRIEVAL Y BÚSQUEDA HÍBRIDA
# ================================
THRESHOLD_CONFIANZA = 0.60
RETRIEVAL_THRESHOLD = 0.50

TOP_K_BM25 = 20
TOP_K_VECTOR = 20

RRF_K = 60
RRF_TOP_N = 15

TOP_K_FINAL = 3

# ================================
# CONFIGURACIÓN DE EMBEDDINGS
# ================================
EMBEDDING_BATCH_SIZE = 32

# ================================
# REINDEXACIÓN GLOBAL
# ================================
FORCE_REINDEX = False

# ================================
# GEMINI API KEY
# ================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
