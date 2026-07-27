import os
import time
import re
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from google import genai
from google.genai import types

import config
from retrieval import ejecutar_retrieval_rag

# =====================================================================
# INICIALIZACIÓN DEL CLIENTE GEMINI
# =====================================================================
def get_gemini_client() -> genai.Client:
    """
    Obtiene el cliente de Gemini priorizando Google Colab userdata.
    Lanza RuntimeError si no se encuentra ninguna clave configurada.
    """
    api_key = None
    
    # Prioridad 1: Google Colab Userdata
    try:
        from google.colab import userdata
        api_key = userdata.get('GEMINI_API_KEY')
    except Exception:
        pass

    # Prioridad 2: Variable de entorno
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    # Fallo rápido si no hay API Key
    if not api_key:
        raise RuntimeError(
            "No se encontró la variable GEMINI_API_KEY en 'userdata' ni en variables de entorno. "
            "Asegúrate de configurarla antes de ejecutar el módulo."
        )

    return genai.Client(api_key=api_key)


# =====================================================================
# CONSTRUCTOR DE PROMPTS
# =====================================================================
class PromptBuilder:
    @staticmethod
    def construir_prompt(pregunta: str, documentos_recuperados: List[Dict[str, Any]]) -> Tuple[str, str]:
        """Construye las instrucciones del sistema y el prompt de usuario con deduplicación de contextos."""
        bloques_contexto = []
        vistos = set()
        
        for d in documentos_recuperados:
            texto = d["metadata"]["texto_raw"].strip()
            if texto in vistos:
                continue
            vistos.add(texto)
            
            meta = d["metadata"]
            pag_cod = meta.get('pagina', meta.get('codigo', 'N/A'))
            ref = f"{meta['tipo_documento']}, Página/Código {pag_cod}"
            bloques_contexto.append(f"--- FUENTE [{ref}] ---\n{texto}")
            
        contexto_unificado = "\n\n".join(bloques_contexto)
        
        prompt_sistema = """Eres un asistente técnico especializado en recubrimientos industriales RIS.

INSTRUCCIONES DE RIGOR TÉCNICO Y FORMATO:
1. Responde únicamente utilizando la información explícita del CONTEXTO TÉCNICO.
2. Para cada afirmación o sección técnica, DEBES citar la fuente utilizando el formato exacto: [Manual Operativo, Página 6] o [Ficha Técnica, Código MAT-01].
3. Si una respuesta requiere combinar varios fragmentos o fuentes, hazlo explícitamente indicando todas las fuentes utilizadas para esa sección.
4. PROHIBIDO UTILIZAR FORMATO LATEX O FÓRMULAS MATEMÁTICAS (No uses $$ o $ para procedimientos ni flechas).
5. Presenta las secuencias, pasos o parámetros mediante listas numeradas o con viñetas en Markdown estándar.
6. Si la información no aparece en el contexto, responde estrictamente: "No dispongo de información suficiente en la documentación técnica para responder a esta pregunta."
7. No utilices conocimiento previo ni asumas datos no documentados."""

        prompt_usuario = f"""CONTEXTO TÉCNICO:
{contexto_unificado}

CONSULTA DEL USUARIO:
{pregunta}

RESPUESTA TÉCNICA:"""

        return prompt_sistema, prompt_usuario


# =====================================================================
# GENERADOR LLM (GEMINI)
# =====================================================================
class LLMGenerator:
    def __init__(self, client_instance: Optional[genai.Client] = None, modelo_fijo: str = config.LLM_MODEL_NAME):
        self.client = client_instance or get_gemini_client()
        self.modelo = modelo_fijo

    def generar(
        self, 
        prompt_sistema: str, 
        prompt_usuario: str, 
        max_intentos: int = getattr(config, 'LLM_MAX_RETRIES', 2)
    ) -> Tuple[Optional[str], str]:
        """Envía la solicitud al modelo Gemini con reintentos para manejo de cuota."""
        for intento in range(max_intentos):
            try:
                response = self.client.models.generate_content(
                    model=self.modelo,
                    contents=prompt_usuario,
                    config=types.GenerateContentConfig(
                        system_instruction=prompt_sistema,
                        temperature=getattr(config, 'LLM_TEMPERATURE', 0.0)
                    )
                )
                if response.text:
                    return response.text.strip(), "ok"
                    
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    tiempo_espera = 10 * (intento + 1)
                    print(f" Ráfaga/Cuota temporal (429). Reintentando en {tiempo_espera}s...")
                    time.sleep(tiempo_espera)
                else:
                    print(f"Error en modelo {self.modelo}: {msg}")
                    break
                    
        return None, "error_api_o_cuota"


# =====================================================================
# VALIDADOR TÉCNICO FLEXIBLE
# =====================================================================
class ResponseValidator:
    @staticmethod
    def validar(respuesta_llm: str, es_confiable_retrieval: bool) -> Tuple[bool, str]:
        """Valida las reglas de negocio y presencia de referencias con un patrón flexible."""
        if not es_confiable_retrieval:
            return False, "retrieval_bajo_umbral"
            
        if len(respuesta_llm.strip()) < 30:
            return False, "respuesta_demasiado_corta"

        frases_evasivas = [
            "no dispongo de información",
            "no se encuentra en los documentos",
            "información insuficiente"
        ]
        if any(f in respuesta_llm.lower() for f in frases_evasivas):
            return False, "informacion_insuficiente"
            
        # Regex optimizado y más flexible: acepta citas con/sin corchetes
        patron_citas = r"(?:\[[^\]]*(?:Manual|Ficha|Página|Código|Pág|MAT|EQ)[^\]]*\]|(?:Manual|Ficha|Página|Código|Pág|MAT|EQ)\s*[\w\-]+)"
        tiene_citas = bool(re.search(patron_citas, respuesta_llm, re.IGNORECASE))
        
        if not tiene_citas:
            return False, "sin_citas_validas"

        return True, "ok"


# =====================================================================
# FORMATEADOR DE RESPUESTAS Y FALLBACKS (DETERMINISTA)
# =====================================================================
class ResponseFormatter:
    @staticmethod
    def obtener_nivel_confianza(score: float) -> str:
        if score >= 0.80:
            return f"Alta ({score:.2f})"
        elif score >= 0.60:
            return f"Media ({score:.2f})"
        else:
            return f"Baja ({score:.2f})"

    @staticmethod
    def generar_fallback(pregunta: str, motivo: str) -> str:
        mensajes = {
            "score_retrieval_bajo": "No se encontraron documentos técnicos con suficiente evidencia en la base de conocimientos para responder esta consulta.",
            "error_api_o_cuota": "La consulta no pudo ser procesada por limitaciones temporales en la API del LLM. Intente de nuevo en un minuto.",
            "sin_citas_validas": "La respuesta generada no incluyó referencias explícitas a la documentación.",
            "informacion_insuficiente": "La información solicitada no está disponible en la documentación técnica actual."
        }
        explicacion = mensajes.get(motivo, "No fue posible procesar la consulta con suficiente evidencia documental.")
        
        return f"""### Consulta No Completada

> **Consulta:** *"{pregunta}"*

{explicacion}

**Sugerencias:**
• Revisa el manual o ficha técnica correspondiente.
• Consulte directamente con el área técnica responsable.
• Reformule la pregunta con términos más específicos.

---
*Estado de Seguridad:* Rechazado (`Motivo: {motivo}`)"""

    @classmethod
    def dar_formato_final(cls, respuesta_llm: str, documentos_recuperados: List[Dict[str, Any]], score_retrieval: float) -> str:
        # Fuentes 100% deterministas basadas en la metadata de retrieval recuperada
        fuentes_limpias = set()
        for d in documentos_recuperados:
            meta = d["metadata"]
            pag_cod = meta.get('pagina', meta.get('codigo', 'N/A'))
            tipo_doc = meta.get('tipo_documento', 'Documento')
            fuentes_limpias.add(f"• {tipo_doc}, Página/Código {pag_cod}")

        bloque_fuentes = "\n".join(sorted(list(fuentes_limpias)))
        nivel_confianza = cls.obtener_nivel_confianza(score_retrieval)

        return f"""{respuesta_llm}

---
### Fuentes Consultadas
{bloque_fuentes}

**Nivel de Confianza del Retrieval:** `{nivel_confianza}`"""


# =====================================================================
# AGENTE RAG PRINCIPAL CON TRAZABILIDAD
# =====================================================================
class RAGAgent:
    def __init__(self, rag_resources: Dict[str, Any]):
        self.rag_resources = rag_resources
        self.prompt_builder = PromptBuilder()
        self.llm_generator = LLMGenerator()
        self.validator = ResponseValidator()
        self.formatter = ResponseFormatter()
        self.historial_evaluacion: List[Dict[str, Any]] = []

    def responder_consulta(self, pregunta: str) -> str:
        inicio_tiempo = time.time()
        
        # 1. Retrieval
        candidatos, score, es_confiable = ejecutar_retrieval_rag(pregunta, self.rag_resources)
        
        ids_chunks = [
            f"{d['metadata'].get('tipo_documento','doc')}_p{d['metadata'].get('pagina', d['metadata'].get('codigo', 'NA'))}"
            for d in candidatos
        ]
        
        if not es_confiable:
            respuesta_final = self.formatter.generar_fallback(pregunta, "score_retrieval_bajo")
            self._registrar_experimento(pregunta, score, "rechazado_retrieval", ids_chunks, time.time() - inicio_tiempo)
            return respuesta_final
            
        # 2. Prompting (Se descarta la variable no usada con '_')
        prompt_sis, prompt_usr = self.prompt_builder.construir_prompt(pregunta, candidatos)
        
        # 3. LLM Generation
        respuesta_raw, estatus_llm = self.llm_generator.generar(prompt_sis, prompt_usr)
        
        if estatus_llm != "ok":
            respuesta_final = self.formatter.generar_fallback(pregunta, estatus_llm)
            self._registrar_experimento(pregunta, score, f"fallo_llm_{estatus_llm}", ids_chunks, time.time() - inicio_tiempo)
            return respuesta_final

        # 4. Validation
        es_valida, motivo = self.validator.validar(respuesta_raw, es_confiable)
        
        # 5. Output
        if es_valida:
            respuesta_final = self.formatter.dar_formato_final(respuesta_raw, candidatos, score)
            self._registrar_experimento(pregunta, score, "éxito", ids_chunks, time.time() - inicio_tiempo)
            return respuesta_final
        else:
            respuesta_final = self.formatter.generar_fallback(pregunta, motivo)
            self._registrar_experimento(pregunta, score, f"rechazado_validación_{motivo}", ids_chunks, time.time() - inicio_tiempo)
            return respuesta_final

    def _registrar_experimento(self, pregunta: str, score: float, estado: str, ids_chunks: List[str], tiempo_ejecucion: float):
        registro = {
            "Fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Modelo_LLM": self.llm_generator.modelo,
            "Pregunta": pregunta,
            "Score_Retrieval": round(score, 4),
            "Nivel_Confianza": self.formatter.obtener_nivel_confianza(score),
            "Chunks_Consultados": ", ".join(ids_chunks),
            "Estado": estado,
            "Tiempo_Segundos": round(tiempo_ejecucion, 2)
        }
        self.historial_evaluacion.append(registro)

    def exportar_historial(self, ruta_csv: str = "historial_evaluacion_rag.csv"):
        df = pd.DataFrame(self.historial_evaluacion)
        df.to_csv(ruta_csv, index=False)
        print(f"Historial guardado exitosamente en: {ruta_csv}")
