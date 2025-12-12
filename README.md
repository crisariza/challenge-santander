# Sistema de Question Answering - Challenge Santander Tecnología

Sistema de Question Answering implementado con embeddings + retrieval para responder consultas de clientes bancarios basándose en respuestas históricas del equipo de soporte.

## Descripción

Se implementa un pipeline completo de IA/ML que:

1. **Carga y procesa** datos de consultas y respuestas de soporte
2. **Genera embeddings** de los documentos
3. **Realiza búsqueda semántica** usando FAISS para encontrar información importante
4. **Genera respuestas** usando un modelo local que extrae respuestas de documentos relevantes

## Cómo ejecutar

### Requisitos Previos

- Python 3.8 o mayor
- pip

### Instalación

1. **Instalar dependencias**:

```bash
pip install -r requirements.txt
```

**Nota**: En la primer ejecución del archivo, se descargará de forma automática el modelo de embeddings (`paraphrase-multilingual-MiniLM-L12-v2`).

### Ejecución

```bash
python qa_system.py
```

## Breve explicación del script

### 1. Inicializar el sistema

Primero se inicializa el QA, esta hecho con un modelo offline, decidi usarlo porque no tenia experienca con modelos locales y me parecia un buen desafío. También es una simplicación ejecutar el challenge ya que no necesita API keys y funciona offline.

### 2. Cargar y Preparar Datos

Se cargan los datos del CSV y se preparan los documentos para el procesamiento (combina pregunta y respuesta, hace chunking si es necesario).

### 3. Generar y guardar embeddings

Se generan todos los embeddings semánticos para los documentos. Puede llevar un tiempo.
Todos los embeddings los guardamos para poder reutiliarlos nuevamente sin tener que regenerarlos en cada ejecución del script.

### 4. Realizar Consultas

Para simplificar el challenge, tenemos estas preguntas pre-hechas:
test_queries = [
"¿Cómo cambio mi dirección?",
"Mi tarjeta está bloqueada",
"¿Cuánto tarda un reclamo?",
"¿Cómo compro dólares?"
] (línea 211 del script)

Me parecio más practico que obtener una query por parametros, asi solamente con ejecutar el script estan las preguntas. Se pueden agregar, sacar, modificar, etc para probar las respuestas con otras preguntas.

## Explicaciones tecnicas

**Modelo de Embeddings**: Se eligió `paraphrase-multilingual-MiniLM-L12-v2` porque es multilingüe (soporta español), ligero, rápido, no requiere GPU y funciona offline. Modelos mas grandes serian demasiado para un challenge de esta magnitud.

**Preprocesamiento de Datos**: Se combinaron preguntas y respuestas en un solo documento (`"Pregunta: {query}\nRespuesta: {response}"`) para proporcionar contexto completo en la búsqueda semántica. El archivo original del csv fue ligeramente corregido en los problemas de encoding que tenia.

**Chunking Inteligente**:

- Solo se aplica a documentos mayores a 300 caracteres para evitar fragmentación innecesaria.
- Se usa overlap de 50 caracteres: cuando se divide un texto, cada chunk se superpone con el anterior en 50 caracteres. Por ejemplo, si un texto se divide en chunks de 300 caracteres, el segundo chunk empieza en el carácter 250 (y no en el 300), repitiendo los últimos 50 caracteres del chunk anterior. Esto evita perder información importante que podría quedar cortada entre dos chunks.
- El corte se hace en puntos naturales (puntos, comas, espacios) y solo si el punto está en la segunda mitad del chunk (>50%) para evitar fragmentación excesiva.

**Búsqueda Semántica con FAISS**: Se utilizó `IndexFlatL2` (búsqueda exacta con distancia L2) porque es muy rápido para datasets pequeños/medianos, fácil de usar, no requiere configuración compleja y permite guardar/cargar índices. Se descartaron ChromaDB (overhead innecesario), bases de datos vectoriales cloud (en mi opinión, overkill para este dataset) y listas en memoria (no escalables). La distancia L2 funciona bien con embeddings normalizados.

**Almacenamiento**: Se guardan el índice FAISS y metadatos en disco (formato nativo FAISS para el índice, pickle para documentos/metadatos) para reutilizar embeddings sin regenerarlos en cada ejecución. Esto acelera fuertemente cuando se vuelva a correr el script.

**Estructura Interna**: El script tiene dos listas paralelas: `self.documents` (textos originales o chunks) y `self.metadata` (id, category, original_query, original_response, created_at) para mantener trazabilidad completa de cada documento.

## Mejoras posibles

Si tuviera más tiempo para desarrollar, implementaría estas mejoras:

- Métricas como BLEU, ROUGE y Exact Match para medir la calidad de las respuestas
- Tokenización por oraciones completas en lugar de cortes por caracteres
- Permitir filtrar búsquedas por tipo (cuenta, tarjeta, etc)
- Combinar búsqueda semántica con keywords para términos específicos
- Integrar Llama 2 o Mistral para generar respuestas más naturales
- Migrar a IndexIVF o IndexHNSW en el caso de tener datasets más grandes (lo mas común para escalar)
- Implementar caché para consultas frecuentes
