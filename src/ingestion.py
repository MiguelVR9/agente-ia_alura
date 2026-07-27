import os
import re
import pandas as pd
import pypdf
from typing import List, Tuple, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

# ================================
# VALIDACIÓN DE ARCHIVOS
# ================================
def validate_file_path(file_path: str) -> None:
    """Verifica si el archivo existe antes de intentar cargarlo."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"❌ Archivo no encontrado: {file_path}. "
            f"Verifica que el archivo esté presente en la carpeta '{config.DATA_DIR}'."
        )

# ================================
# TOKENIZADOR TÉCNICO UNIFICADO
# ================================
def tokenize_technical_text(text: str) -> List[str]:
    """Preserva acentos, números, caracteres especiales y guiones en términos técnicos."""
    return re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\-_/\.]+", text.lower())

# ================================
# DICCIONARIO DE CONCEPTOS EQUIVALENTES
# ================================
DICCIONARIO_EXPANSION = {
    "granallado": ["chorro abrasivo", "abrasivo", "sa2.5", "sspc-sp10"],
    "epoxi": ["epoxico", "epo"],
    "pelicula": ["espesor", "mils", "micras", "micronage"],
    "limpieza": ["preparacion", "acondicionamiento"]
}

def expandir_query(query: str) -> str:
    """Enriquece la consulta expandiendo tokens con sus equivalentes técnicos."""
    tokens = tokenize_technical_text(query)
    query_expandida = []
    
    for token in tokens:
        query_expandida.append(token)
        if token in DICCIONARIO_EXPANSION:
            query_expandida.extend(DICCIONARIO_EXPANSION[token])
            
    return " ".join(query_expandida)

# ================================
# CARGA DE PDFs CON ENRIQUECIMIENTO CONTEXTUAL
# ================================
def load_and_chunk_pdf(
    pdf_path: str, 
    doc_type: str, 
    ignore_top_pages: int = 0, 
    chunk_size: int = config.CHUNK_SIZE, 
    chunk_overlap: int = config.CHUNK_OVERLAP
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Carga PDF usando pypdf, divide en chunks y genera metadatos enriquecidos."""
    validate_file_path(pdf_path)

    try:
        reader = pypdf.PdfReader(pdf_path)
    except Exception as e:
        print(f"Error al abrir PDF {pdf_path}: {e}")
        return [], []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    metadatos = []
    chunk_counter = 0
    prefix = doc_type.lower().replace(" ", "_")
    
    for idx in range(ignore_top_pages, len(reader.pages)):
        page_text = reader.pages[idx].extract_text()
        if not page_text or len(page_text.strip()) < 50:
            continue
            
        page_chunks = text_splitter.split_text(page_text)
        
        for c in page_chunks:
            chunk_counter += 1
            unique_chunk_id = f"{prefix}_{chunk_counter}"
            text_con_contexto = f"Documento: {doc_type} | Página: {idx + 1}\nContenido:\n{c}"
            
            chunks.append(text_con_contexto)
            metadatos.append({
                "chunk_id": unique_chunk_id,
                "fuente": pdf_path,
                "tipo_documento": doc_type,
                "pagina": idx + 1,
                "texto_raw": c
            })
            
    return chunks, metadatos

# ================================
# CARGA DE EXCEL CON METADATOS
# ================================
def load_excel_as_chunks(
    excel_path: str, 
    entity_type: str
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Carga archivos Excel y convierte cada fila en un chunk enriquecido."""
    validate_file_path(excel_path)

    try:
        xls = pd.ExcelFile(excel_path)
        df = pd.read_excel(excel_path, sheet_name=xls.sheet_names[0])
    except Exception as e:
        print(f"Error al abrir Excel {excel_path}: {e}")
        return [], []
    
    chunks = []
    metadatos = []
    chunk_counter = 0
    
    for idx, row in df.iterrows():
        chunk_counter += 1
        row_str_list = [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
        
        codigo_val = row.get('Código ' + entity_type, idx + 1)
        row_text = f"Ficha de {entity_type} [{codigo_val}]:\n" + " | ".join(row_str_list)
        
        chunks.append(row_text)
        metadatos.append({
            "chunk_id": f"excel_{entity_type.lower()}_{chunk_counter}",
            "fuente": excel_path,
            "tipo_documento": f"Excel {entity_type}",
            "pagina": "N/A",
            "codigo": str(codigo_val),
            "texto_raw": row_text
        })
        
    return chunks, metadatos

# ================================
# FUNCIÓN ORQUESTADORA DE INGESTIÓN COMPLETA
# ================================
def load_all_documents() -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """Ejecuta la carga masiva utilizando las rutas definidas en config.py."""
    chunks_fichas, meta_fichas = load_and_chunk_pdf(
        config.FICHAS_TECNICAS_PATH, "Ficha Técnica", ignore_top_pages=config.IGNORE_TOP_PAGES_FICHAS
    )
    chunks_manuales, meta_manuales = load_and_chunk_pdf(
        config.MANUALES_PATH, "Manual Operativo", ignore_top_pages=config.IGNORE_TOP_PAGES_MANUAL
    )
    chunks_mat_excel, meta_mat_excel = load_excel_as_chunks(
        config.MATERIALES_EXCEL_PATH, "Material"
    )
    chunks_eq_excel, meta_eq_excel = load_excel_as_chunks(
        config.EQUIPOS_EXCEL_PATH, "Equipo"
    )

    todos_los_chunks = chunks_fichas + chunks_manuales + chunks_mat_excel + chunks_eq_excel
    todos_los_metadatos = meta_fichas + meta_manuales + meta_mat_excel + meta_eq_excel
    todos_los_ids = [m["chunk_id"] for m in todos_los_metadatos]

    print(f"✓ Documentos cargados con éxito: {len(todos_los_chunks)} chunks totales generados.")
    return todos_los_chunks, todos_los_metadatos, todos_los_ids

if __name__ == "__main__":
    chunks, metadata, ids = load_all_documents()
    print("\n===== RESUMEN DE INGESTIÓN =====")
    print(f"• Total Chunks: {len(chunks)}")
    print(f"• Total Metadatos: {len(metadata)}")
    print(f"• Total IDs: {len(ids)}")
