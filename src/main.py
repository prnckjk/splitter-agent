import asyncio
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_index.llms.openai import OpenAI
from src.agent.splitter_agent import SplitterAgent
from src.utils.csv_handler import read_csv, write_excel
from src.evaluation.evaluator_agent import EvaluatorAgent

async def main():
    if len(sys.argv) != 2:
        print("Uso: python src/main.py <nombre_archivo_csv_entrada>")
        sys.exit(1)

    input_filename = sys.argv[1]
    output_filename = f"split_{os.path.basename(input_filename).replace('.csv', '.xlsx')}"
    evaluation_filename = f"evaluation_{os.path.basename(input_filename).replace('.csv', '.xlsx')}"

    llm_splitter = OpenAI(temperature=0, model="gpt-4o-mini")
    llm_evaluator = OpenAI(temperature=0, model="gpt-4o")
    
    splitter_agent = SplitterAgent(llm=llm_splitter)
    evaluator_agent = EvaluatorAgent(llm=llm_evaluator)

    input_data = read_csv(input_filename)

    try:
        results = await splitter_agent.process_csv_data(input_data)
        print(f"Total de mensajes procesados: {len(results)}")
        write_excel(results, output_filename)
        
        print("\nEvaluando resultados...")
        evaluation_results = await evaluator_agent.evaluate_batch(results)
        
        write_excel(evaluation_results, evaluation_filename)
        
        # Calcular promedios de puntuaciones
        df_eval = pd.DataFrame(evaluation_results)
        numeric_columns = ['precisión_del_tipo_de_mensaje', 'relevancia_de_la_categoría', 
                           'exhaustividad_del_contenido', 'calidad_de_la_extracción', 'puntuación_media']
        avg_scores = df_eval[numeric_columns].mean()
        
        print("\nPuntuaciones promedio de evaluación:")
        for metric, score in avg_scores.items():
            print(f"{metric.replace('_', ' ').capitalize()}: {score:.2f}")
            if score < 4:
                print(f"ATENCIÓN: La puntuación de {metric} está por debajo de 4.")
        
    except Exception as e:
        print(f"Ocurrió un error: {e}")
    finally:
        splitter_agent.reset()

if __name__ == "__main__":
    asyncio.run(main())