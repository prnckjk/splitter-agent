# Proyecto SplitterAgent: Análisis y Segmentación de Consultas de Usuarios

## Índice
1. [Contexto y Objetivo](#contexto-y-objetivo)
2. [Análisis de Datos](#análisis-de-datos)
3. [Diseño del Sistema](#diseño-del-sistema)
4. [Implementación](#implementación)
   - [SplitterAgent](#splitteragent)
   - [EvaluatorAgent](#evaluatoragent)
5. [Evaluación](#evaluación)
6. [Escalabilidad y Mejoras](#escalabilidad-y-mejoras)
7. [Instrucciones de Ejecución](#instrucciones-de-ejecución)

## Contexto y Objetivo

En TaxDown, utilizamos modelos de lenguaje avanzados (LLMs) para procesar y responder a las consultas de nuestros usuarios. Un paso crucial en nuestro pipeline de generación de respuestas es el "splitter", un componente que se encarga de analizar los mensajes de los usuarios (que pueden llegar por chat o email) y extraer las distintas preguntas o temas que plantean para poder contestarlos por separado.

El objetivo principal es diseñar e implementar un sistema de "splitter" que pueda:
- Identificar y extraer múltiples preguntas o temas de un solo mensaje de usuario.
- Mantener juntas las frases relacionadas con cada pregunta, incluyendo el contexto adicional o la elaboración proporcionada por el usuario.
- Manejar eficazmente tanto mensajes de chat como emails, que pueden tener estructuras y longitudes diferentes.

## Análisis de Datos

Tras examinar el dataset proporcionado, se identificaron los siguientes patrones comunes:

1. Los usuarios tienden a formular múltiples preguntas en un solo mensaje.
2. A menudo, se proporciona contexto adicional o se elabora sobre una pregunta después de haberla planteado.
3. Los mensajes varían significativamente en longitud y estructura, especialmente entre chats y emails.
4. Muchos mensajes contienen información personal o financiera específica que requiere un manejo cuidadoso.

## Diseño del Sistema

Para abordar este desafío, se ha optado por utilizar un enfoque basado en un modelo de lenguaje grande (LLM) con fine-tuning específico para la tarea de segmentación. Este enfoque se considera el más adecuado por las siguientes razones:

1. **Flexibilidad**: Los LLMs pueden adaptarse a una amplia variedad de estructuras de mensajes y estilos de escritura.
2. **Comprensión contextual**: Pueden entender y mantener el contexto a lo largo de mensajes largos.
3. **Capacidad de generalización**: Pueden manejar casos nuevos o inusuales que no se hayan visto en los datos de entrenamiento.

### Comparación de Técnicas

| Factor | LLM con Fine-tuning | Enfoque basado en reglas | Modelo de clasificación tradicional |
|--------|---------------------|--------------------------|-------------------------------------|
| Coste computacional y financiero | Alto | Bajo | Medio |
| Latencia | Media | Baja | Baja |
| Calidad de los resultados | Alta | Media | Media |
| Escalabilidad | Alta | Baja | Media |
| Mantenibilidad | Media | Baja | Media |
| Flexibilidad | Alta | Baja | Media |
| Requisitos de datos de entrenamiento | Altos | Bajos | Medios |
| Interpretabilidad del modelo | Baja | Alta | Media |

## Implementación

La implementación se ha realizado utilizando el framework LlamaIndex y la API de OpenAI. El sistema consta de dos componentes principales: el SplitterAgent y el EvaluatorAgent.

### SplitterAgent

El SplitterAgent es responsable de procesar los mensajes de los usuarios y dividirlos en preguntas o temas distintos. 

#### Proceso de división del mensaje original:

1. **Análisis del mensaje**: El agente analiza el mensaje completo para identificar su estructura y contenido.

2. **Identificación del tipo de mensaje**: Se determina si el mensaje es una consulta, una comunicación ordinaria, un agradecimiento, etc.

3. **Extracción de categorías**: El agente identifica las diferentes categorías o temas presentes en el mensaje. Estas pueden ser áreas como "Declaración de impuestos", "Deducciones", "Plazos", etc.

4. **Segmentación del contenido**: Para cada categoría identificada, el agente extrae las partes relevantes del mensaje original, incluyendo preguntas específicas, contexto relacionado y cualquier información adicional proporcionada por el usuario.

5. **Estructuración de la salida**: El resultado se estructura en un formato JSON que incluye:
   - El mensaje original completo
   - El tipo de mensaje identificado
   - Una lista de categorías, cada una con su propio contenido extraído

Ejemplo de salida del SplitterAgent:

```json
{
  "original_message": "Hola, tengo dos preguntas. Primero, ¿cuándo es la fecha límite para presentar la declaración? Segundo, ¿puedo deducir los gastos de mi oficina en casa?",
  "message_type": "consulta",
  "categories": [
    {
      "keyword": "Plazos",
      "content": ["¿Cuándo es la fecha límite para presentar la declaración?"]
    },
    {
      "keyword": "Deducciones",
      "content": ["¿Puedo deducir los gastos de mi oficina en casa?"]
    }
  ]
}
```

Implementación del SplitterAgent:

```python
class SplitterAgent:
    def __init__(self, llm: OpenAI = OpenAI(temperature=0, model="gpt-4o-mini")):
        self._llm = llm
        self._instructions = self._load_instructions()

    def _process_message(self, message: str) -> dict:
        self._chat_history.append(ChatMessage(role="user", content=message))
        
        system_message = ChatMessage(role="system", content=self._instructions)
        messages = [system_message] + self._chat_history

        try:
            ai_message = self._llm.chat(messages).message
            result = json.loads(ai_message.content)
            result['original_message'] = message
            return result
        except json.JSONDecodeError as e:
            print(f"Error de decodificación JSON: {e}")
            return {
                "error": "Respuesta JSON inválida",
                "raw_content": ai_message.content,
                "original_message": message
            }
```

### EvaluatorAgent

El EvaluatorAgent se encarga de evaluar la calidad de las divisiones realizadas por el SplitterAgent. 

#### Proceso de evaluación de las extracciones:

1. **Análisis holístico**: El EvaluatorAgent examina todas las extracciones en conjunto, considerando el mensaje original completo y todas las categorías y contenidos extraídos.

2. **Evaluación de criterios múltiples**: Para cada conjunto de extracciones, el agente evalúa:
   - Precisión del tipo de mensaje: ¿Se ha identificado correctamente si es una consulta, comunicación, etc.?
   - Relevancia de las categorías: ¿Las categorías identificadas son apropiadas para el contenido del mensaje?
   - Exhaustividad del contenido: ¿Se han capturado todos los temas y preguntas importantes del mensaje original?
   - Calidad de la extracción: ¿El contenido extraído refleja con precisión la intención y el contexto del mensaje original?

3. **Puntuación**: Cada criterio se puntúa en una escala del 1 al 5, donde 5 es la mejor puntuación.

4. **Comentario cualitativo**: Además de las puntuaciones numéricas, el evaluador proporciona un comentario detallado explicando sus evaluaciones y sugiriendo posibles mejoras.

5. **Cálculo de puntuación media**: Se calcula una puntuación media basada en los criterios individuales para dar una evaluación general de la calidad de la división.

Ejemplo de salida del EvaluatorAgent:

```json
{
  "precisión_del_tipo_de_mensaje": 5.0,
  "relevancia_de_la_categoría": 4.5,
  "exhaustividad_del_contenido": 4.0,
  "calidad_de_la_extracción": 4.5,
  "comentario": "La división del mensaje es generalmente buena. El tipo de mensaje se identificó correctamente como una consulta. Las categorías 'Plazos' y 'Deducciones' son relevantes y capturan los temas principales. Sin embargo, se podría mejorar la exhaustividad incluyendo más contexto sobre la situación de trabajo desde casa en la categoría de deducciones.",
  "puntuación_media": 4.5,
  "mensaje_original": "Hola, tengo dos preguntas. Primero, ¿cuándo es la fecha límite para presentar la declaración? Segundo, ¿puedo deducir los gastos de mi oficina en casa?"
}
```

Implementación del EvaluatorAgent:

```python
class EvaluatorAgent:
    def __init__(self, llm: OpenAI = OpenAI(temperature=0, model="gpt-4o")):
        self._llm = llm
        self._instructions = self._load_instructions()

    def evaluate(self, split_result: Dict) -> Dict:
        evaluation_text = f"""
        Mensaje Original: {split_result['original_message']}
        Tipo de Mensaje: {split_result['message_type']}
        Categorías y Contenidos:
        {json.dumps(split_result['categories'], indent=2, ensure_ascii=False)}
        """
        
        return self._process_evaluation(evaluation_text, split_result['original_message'])

    def _process_evaluation(self, evaluation_text: str, original_message: str) -> Dict:
        try:
            ai_message = self._llm.chat([ChatMessage(role="system", content=self._instructions),
                                         ChatMessage(role="user", content=evaluation_text)]).message
            
            result = json.loads(re.sub(r'```json\s*|\s*```', '', ai_message.content))
            
            required_keys = ["precisión_del_tipo_de_mensaje", "relevancia_de_la_categoría", 
                             "exhaustividad_del_contenido", "calidad_de_la_extracción", "comentario"]
            
            for key in required_keys:
                if key not in result:
                    raise ValueError(f"Falta la clave requerida: {key}")
                if key != "comentario":
                    result[key] = float(result[key])
            
            result["puntuación_media"] = sum(result[key] for key in required_keys if key != "comentario") / 4
            result["mensaje_original"] = original_message
            
            return result
        except Exception as e:
            return self._generate_error_response(str(e), original_message)
```

Este proceso de dos etapas (splitting y evaluación) permite no solo dividir los mensajes de los usuarios en temas y preguntas relevantes, sino también evaluar la calidad de estas divisiones. Esto proporciona una retroalimentación valiosa sobre el rendimiento del sistema y permite identificar áreas de mejora.

## Evaluación

Para evaluar el sistema de manera automatizada, se ha implementado un enfoque basado en la comparación semántica en lugar de la coincidencia exacta de palabras. El EvaluatorAgent utiliza un modelo de lenguaje para evaluar la calidad de las divisiones basándose en criterios predefinidos.

Este enfoque tiene las siguientes ventajas:

1. Permite evaluar la calidad semántica de las divisiones, no solo la correspondencia léxica.
2. Es flexible ante reformulaciones o parafraseos de las preguntas originales.
3. Puede adaptarse a diferentes estilos de escritura y estructuras de mensajes.

La evaluación holística proporcionada por el EvaluatorAgent permite detectar problemas como:

- Omisión de temas o preguntas importantes
- Duplicación innecesaria de información entre categorías
- Inconsistencias en la categorización de temas similares
- Pérdida de contexto importante al dividir el mensaje

## Escalabilidad y Mejoras

Para escalar la solución y manejar un gran volumen de mensajes en tiempo real, se podrían considerar las siguientes estrategias:

1. **Implementación de un sistema de colas**: Utilizar tecnologías como Apache Kafka o RabbitMQ para manejar grandes volúmenes de mensajes entrantes.
2. **Paralelización**: Implementar procesamiento paralelo para manejar múltiples mensajes simultáneamente.
3. **Caching**: Implementar un sistema de caché para almacenar resultados de consultas frecuentes y reducir la carga en el modelo.

Mejoras potenciales para futuras iteraciones:

1. **Fine-tuning específico del dominio**: Entrenar el modelo con datos específicos de TaxDown para mejorar la precisión en el contexto fiscal.
2. **Implementación de técnicas de few-shot learning**: Incorporar ejemplos relevantes en el prompt para mejorar el rendimiento en casos específicos.
3. **Integración de un sistema de retroalimentación del usuario**: Permitir que los usuarios proporcionen feedback sobre la calidad de las divisiones para mejorar continuamente el sistema.

## Instrucciones de Ejecución

Para ejecutar el sistema, sigue estos pasos:

1. Clona el repositorio:
   ```
   git clone https://github.com/prnckjk/splitter-agent.git
   ```

2. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

3. Configura las variables de entorno en un archivo `.env`:
   ```
   OPENAI_API_KEY=tu_clave_api_aqui
   # INPUT/OUTPUT DIRECTORIES
   INPUT_DIRECTORY=tu_ruta_entrada (Opcional)
   OUTPUT_DIRECTORY=tu_ruta_de_salida
   ```

4. Ejecuta el script principal:
   ```
   python src/main.py ruta/al/archivo/de/entrada.csv
   ```

El sistema procesará el archivo de entrada, realizará las divisiones y evaluaciones, y generará dos archivos de salida: uno con las divisiones y otro con las evaluaciones.