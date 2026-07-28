import sys
import os

# 1. Print inmediato para confirmar que Render ejecuta app.py
print("🚀 [LOG 1/5] Iniciando ejecución de app.py...", flush=True)

import gradio as gr
from dotenv import load_dotenv

print("📦 [LOG 2/5] Cargando variables de entorno...", flush=True)
load_dotenv()

# Asegurar que 'src/' sea la primera ruta prioritaria para importaciones
SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, SRC_DIR)

print("📦 [LOG 3/5] Importando módulos locales RAG (indexer y generator)...", flush=True)

try:
    from indexer import initialize_indexes
    from generator import RAGAgent
    print("✅ [LOG 4/5] Módulos locales importados exitosamente.", flush=True)
except Exception as e:
    print(f"❌ [ERROR CRÍTICO] Falló la importación de módulos: {e}", flush=True)
    raise e

# Instancia global para Lazy Loading
agente_ris = None

def obtener_agente():
    """Inicializa el RAG de forma perezosa para evitar timeouts de arranque."""
    global agente_ris
    if agente_ris is None:
        print("⚙️ Inicializando índices y modelos del RAG...", flush=True)
        try:
            rag_resources = initialize_indexes()
            agente_ris = RAGAgent(rag_resources)
            print("✅ Agente RAG listo para consultas.", flush=True)
        except Exception as e:
            print(f"❌ Error crítico al inicializar el RAG: {e}", flush=True)
            raise e
    return agente_ris

def responder_chat(mensaje, historial):
    """Procesa la consulta ingresada por el usuario."""
    if not mensaje or not mensaje.strip():
        return "Por favor, ingresa una consulta válida."
    
    try:
        agente = obtener_agente()
        respuesta = agente.responder_consulta(mensaje)
        return respuesta
    except Exception as e:
        return f"⚠️ Ocurrió un error al procesar tu consulta: {str(e)}"

# Interfaz de Gradio
demo = gr.ChatInterface(
    fn=responder_chat,
    title="Asistente Técnico RIS (RAG)",
    description="Haz tus consultas sobre fichas técnicas, manuales y especificaciones de equipos de RIS.",
    examples=[
        "¿Cuál es el procedimiento y preparación para corrosión severa con chorro de arena?",
        "¿Cuál es el procedimiento para solicitar viáticos de viaje?"
    ]
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"🌐 [LOG 5/5] Levantando servidor Gradio en 0.0.0.0:{port}...", flush=True)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
