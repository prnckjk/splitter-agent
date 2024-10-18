import pandas as pd
from typing import List
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
import tiktoken
import os
import json

MAX_TOKENS = 4000

class SplitterAgent:
    def __init__(
        self,
        llm: OpenAI = OpenAI(temperature=0, model="gpt-4o-mini"),
        chat_history: List[ChatMessage] = [],
    ) -> None:
        self._llm = llm
        self._chat_history = chat_history
        self._tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        self._instructions = self._load_instructions()

    def _load_instructions(self) -> str:
        instructions_path = os.path.join(
            os.path.dirname(__file__),
            "prompts",
            "splitter_instructions.txt"
        )
        with open(instructions_path, 'r', encoding='utf-8') as file:
            return file.read().strip()

    def reset(self) -> None:
        self._chat_history = []

    def num_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def chat(self, message: str) -> dict:
        tokens = self.num_tokens(message)
        if tokens > MAX_TOKENS:
            print(f"Message too long ({tokens} tokens). Splitting...")
            parts = []
            while message:
                part = message[:MAX_TOKENS]
                parts.append(part)
                message = message[MAX_TOKENS:]
            
            results = []
            for part in parts:
                try:
                    result = self._process_message(part)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing part: {e}")
            return self._combine_results(results)
        else:
            return self._process_message(message)

    def _process_message(self, message: str) -> dict:
        """
        Procesa un mensaje utilizando el modelo de lenguaje y extrae la información relevante.

        Args:
            message (str): El mensaje original a procesar.

        Returns:
            dict: Un diccionario con la información extraída y procesada del mensaje.
        """
        self._chat_history.append(ChatMessage(role="user", content=message))
        
        system_message = ChatMessage(role="system", content=self._instructions)
        messages = [system_message] + self._chat_history

        try:
            ai_message = self._llm.chat(messages).message
            self._chat_history.append(ai_message)
            print("Respuesta AI cruda:", ai_message.content)
            try:
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
        except Exception as e:
            print(f"Error en llamada a API: {e}")
            return {
                "error": str(e),
                "original_message": message
            }

    async def process_csv_data(self, data: pd.DataFrame):
        """Process CSV data and split messages."""
        results = []
        text_column = data.select_dtypes(include=['object']).columns[0]
        print(f"Processing column: {text_column}")
        
        for index, row in data.iterrows():
            original_message = row[text_column]
            print(f"\nProcessing message {index+1}/{len(data)}: {original_message[:50]}...")
            try:
                split_result = self.chat(original_message)
                results.append(split_result)
            except Exception as e:
                print(f"Error processing message {index+1}: {e}")
                results.append({
                    "original_message": original_message,
                    "error": str(e)
                })
            self.reset()
        return results