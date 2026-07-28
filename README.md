# Agente de IA corporativo - Recubrimientos Industriales S.A (RIS)

Este proyecto implementa un **Sistema de Generación Aumentada por Recuperación (RAG)** diseñado para responder consultas de los colaboradores de una empresa de recubrimientos industriales ficticia. El agente procesa e interpreta fichas técnicas, manuales de procesos y especificaciones de materiales y equipos para ofrecer respuestas precisas basadas en documentación interna, las cuales fueron generadas por IA a modo de ejemplo para el proyecto.

---

## Arquitectura del Repositorio

```text
agente-ia_alura/
├── data/
│   ├── ejemplos/
│   ├── Especificaciones_Equipos_RIS.xlsx
│   ├── Especificaciones_Materiales_RIS.xlsx
│   ├── Fichas_Tecnicas_Materiales_RIS.pdf
│   └── Manuales_Procesos_Procedimientos_RIS.pdf
├── notebooks/
│   ├── Agente_ia_alura.ipynb
│   └── generador_modulos.ipynb
├── src/
│   ├── config.py          # Configuración de variables globales y entorno
│   ├── ingestion.py       # Lectura, extracción y chunking de documentos (PDF/Excel)
│   ├── indexer.py         # Creación y gestión de índices vectoriales (ChromaDB)
│   ├── retrieval.py       # Motor de búsqueda híbrida y recuperación de contexto (BM25 + Dense)
│   └── generator.py       # Integración con Google Gemini API y orquestación de prompts
├── .env.example           # Plantilla de variables de entorno
├── .gitignore             # Archivos excluidos del control de versiones
├── app.py                 # Interfaz de usuario conversacional con Gradio y Lazy Loading
├── requirements.txt       # Dependencias del proyecto optimizadas para CPU
├── runtime.txt            # Especificación de versión de Python para producción
└── README.md              # Documentación general del proyecto

## 🚀 Fases del Proyecto: De la Idea al Despliegue

### 1. Creación de Datos Técnicos Sintéticos (`/data`)
Para simular un entorno corporativo de recubrimientos industriales sin exponer datos confidenciales, se generaron documentos técnicos sintéticos en formatos PDF y Excel:

* **Fichas Técnicas de Materiales y Equipos:** Propiedades químicas, tiempos de secado, rendimientos y métodos de aplicación.
* **Manuales de Procesos y Procedimientos:** Protocolos de preparación de superficie (limpieza abrasiva, chorro de arena), seguridad y normativas internas (ej. gestión de viáticos).

---

### 2. Experimentación y Validación (`/notebooks`)
* **`Agente_ia_alura.ipynb`:** Espacio de prueba donde se validaron el flujo de datos, los algoritmos de partición de texto (*chunking*), el comportamiento de la búsqueda vectorial y la calidad de respuesta de Google Gemini.
* **`generador_modulos.ipynb`:** Cuaderno utilizado para refactorizar la lógica validada y estructurar los módulos que posteriormente conformarían la arquitectura del sistema.

---

### 3. Modularización del Sistema (`/src`)
El código se desacopló en una arquitectura modular limpia para garantizar mantenibilidad:

* **`config.py`:** Gestión de rutas, claves de API y parámetros globales.
* **`ingestion.py`:** Procesamiento y fragmentación de documentos en `data/`.
* **`indexer.py`:** Indexación de embeddings vectoriales en base de datos local **ChromaDB**.
* **`retrieval.py`:** Algoritmos de recuperación híbrida (semántica mediante vectores y léxica con `rank-bm25`).
* **`generator.py`:** Formateo de prompts y comunicación directa con el modelo de lenguaje de Google Gemini.

---

### 4. Interfaz de Usuario e Integración (`app.py`)
Se construyó una interfaz interactiva tipo chat utilizando **Gradio**.

* **Optimizaciones de Memoria:** Se implementó una estrategia de **Carga Diferida (*Lazy Loading*)**, permitiendo que la interfaz inicie de forma instantánea y que las librerías pesadas e índices se carguen en memoria únicamente cuando el usuario realiza la primera interacción.

---

### 5. Despliegue en la Nube (`Render`)
El repositorio fue conectado a **Render Web Services** para integración y despliegue continuos (CI/CD):

* **Ajuste de Entorno:** Se configuró el archivo `requirements.txt` con PyTorch versión CPU (`torch>=2.4.0`) para mantener la compatibilidad con `transformers` y `sentence-transformers`.
* **Estado de la PoC:** La aplicación se encuentra desplegada y configurada correctamente en el puerto `10000`.

> 💡 **Nota Técnica de Infraestructura:** Este proyecto se entrega como una **Prueba de Concepto (PoC) funcional**. Debido a los límites de memoria de la capa gratuita de Render (512 MB RAM), el procesamiento local de embeddings en CPU mediante `sentence-transformers` puede alcanzar el umbral máximo de memoria en consultas complejas (*OOM*). Para un entorno de producción masivo, se recomienda migrar a un plan con mayor RAM o delegar la generación de embeddings a la API pública de Gemini.

## 🛠️ Guía de Instalación y Ejecución Local

Para ejecutar este proyecto localmente en tu máquina, sigue estos pasos:

### 1. Clonar el repositorio
```bash
git clone [https://github.com/MiguelVR9/agente-ia_alura.git](https://github.com/MiguelVR9/agente-ia_alura.git)
cd agente-ia_alura

### 2. Crear y activar un entorno virtual
# En Linux/macOS:
python3 -m venv venv
source venv/bin/activate

# En Windows:
python -m venv venv
venv\Scripts\activate

### 3.Instalar las dependencias
pip install -r requirements.txt

### 4. Configurar variables de entorno
Crea un archivo .env en la raíz del proyecto tomando como base .env.example:
GEMINI_API_KEY=tu_clave_de_api_de_google_gemini

### 5. Iniciar la aplicación
python app.py
Abre tu navegador e ingresa a http://localhost:7860 (o al puerto indicado en la consola) para interactuar con el asistente.

