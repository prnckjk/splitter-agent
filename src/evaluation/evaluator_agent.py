from typing import List, Dict
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
import tiktoken
import os
import json
import re

MAX_TOKENS = 4000

class EvaluatorAgent:
    """Agente para evaluar la calidad de las divisiones de mensajes."""

    def __init__(
        self,
        llm: OpenAI = OpenAI(temperature=0, model="gpt-4o"),
        chat_history: List[ChatMessage] = [],
    ) -> None:
        """
        Inicializa el agente evaluador.

        Args:
            llm (OpenAI): Modelo de lenguaje a utilizar.
            chat_history (List[ChatMessage]): Historial de chat inicial.
        """
        self._llm = llm
        self._chat_history = chat_history
        self._tokenizer = tiktoken.encoding_for_model("gpt-4o")
        self._instructions = self._load_instructions()

    def _load_instructions(self) -> str:
        """Carga las instrucciones para el evaluador desde un archivo."""
        instructions_path = os.path.join(
            os.path.dirname(__file__),
            "prompts",
            "evaluator_instructions.txt"
        )
        with open(instructions_path, 'r', encoding='utf-8') as file:
            return file.read().strip()

    def reset(self) -> None:
        """Reinicia el historial de chat."""
        self._chat_history = []

    def num_tokens(self, text: str) -> int:
        """Calcula el número de tokens en un texto dado."""
        return len(self._tokenizer.encode(text))

    def evaluate(self, split_result: Dict) -> Dict:
        """
        Evalúa el resultado completo de la división de un mensaje.

        Args:
            split_result (Dict): Resultado de la división de un mensaje.

        Returns:
            Dict: Resultado de la evaluación.
        """
        evaluation_text = f"""
        Mensaje Original: {split_result['original_message']}
        Tipo de Mensaje: {split_result['message_type']}
        Categorías y Contenidos:
        {json.dumps(split_result['categories'], indent=2, ensure_ascii=False)}
        """
        
        return self._process_evaluation(evaluation_text, split_result['original_message'])

    def _process_evaluation(self, evaluation_text: str, original_message: str) -> Dict:
        """Procesa la evaluación utilizando el modelo de lenguaje."""
        self._chat_history.append(ChatMessage(role="user", content=evaluation_text))
        
        system_message = ChatMessage(role="system", content=self._instructions)
        messages = [system_message] + self._chat_history

        try:
            ai_message = self._llm.chat(messages).message
            self._chat_history.append(ai_message)
            
            cleaned_content = re.sub(r'```json\s*|\s*```', '', ai_message.content)
            
            try:
                result = json.loads(cleaned_content)
                required_keys = ["precisión_del_tipo_de_mensaje", "relevancia_de_la_categoría", 
                                 "exhaustividad_del_contenido", "calidad_de_la_extracción", "comentario"]
                for key in required_keys:
                    if key not in result:
                        raise ValueError(f"Falta la clave requerida: {key}")
                
                for key in required_keys:
                    if key != "comentario":
                        result[key] = float(result[key])
                
                scores = [result[key] for key in required_keys if key != "comentario"]
                result["puntuación_media"] = sum(scores) / len(scores)
                result["mensaje_original"] = original_message
                
                return result
            except json.JSONDecodeError as e:
                return self._generate_error_response("Respuesta JSON inválida", original_message)
            except ValueError as e:
                return self._generate_error_response(str(e), original_message)
        except Exception as e:
            return self._generate_error_response(str(e), original_message)

    def _generate_error_response(self, error_message: str, original_message: str) -> Dict:
        """Genera una respuesta de error cuando la evaluación falla."""
        return {
            "precisión_del_tipo_de_mensaje": 0.0,
            "relevancia_de_la_categoría": 0.0,
            "exhaustividad_del_contenido": 0.0,
            "calidad_de_la_extracción": 0.0,
            "comentario": f"Error en la evaluación: {error_message}",
            "mensaje_original": original_message,
            "puntuación_media": 0.0
        }

    async def evaluate_batch(self, data: List[Dict]) -> List[Dict]:
        """Evalúa un lote de resultados de división de mensajes."""
        results = []
        for index, split_result in enumerate(data):
            try:
                if isinstance(split_result, dict) and 'original_message' in split_result:
                    evaluation = self.evaluate(split_result)
                    results.append(evaluation)
                    print(f"Evaluación {index + 1}: Puntuación media = {evaluation['puntuación_media']:.2f}")
                else:
                    raise ValueError(f"Formato de datos inesperado en el índice {index}")
            except Exception as e:
                error_response = self._generate_error_response(str(e), split_result.get('original_message', 'Mensaje no disponible'))
                results.append(error_response)
                print(f"Evaluación {index + 1}: Error - {str(e)}")
            self.reset()
        return results