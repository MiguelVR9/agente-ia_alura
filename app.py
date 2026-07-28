import sys
import os
import gradio as gr
from dotenv import load_dotenv

print("🚀 Iniciando aplicación Gradio...", flush=True)
load_dotenv()

SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, SRC_DIR)

# Variable global para mantener la instancia del RAG en memoria
agente_ris = None

def obtener_agente():
    """Carga los módulos pesados y el RAG solo cuando se hace la primera consulta."""
    global agente_ris
    if agente_ris is None:
        print("⚙️ Cargando librerías e índices por primera vez (Lazy Load)...", flush=True)
        try:
            # Importaciones diferidas dentro de la función
            from indexer import initialize_indexes
            from generator import RAGAgent
            
            rag_resources = initialize_indexes()
            agente_ris = RAGAgent(rag_resources)
            print("✅ Agente RAG listo para responder.", flush=True)
        except Exception as e:
            print(f"❌ Error al inicializar el RAG: {e}", flush=True)
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

# Interfaz de Gradio ultraligera al arrancar
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
    print(f"🌐 Levantando interfaz en el puerto {port}...", flush=True)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
