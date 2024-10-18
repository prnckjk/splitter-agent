import pandas as pd
import os
from src.config import INPUT_DIRECTORY, OUTPUT_DIRECTORY


def read_csv(filename: str) -> pd.DataFrame:
    """
    Lee un archivo CSV y devuelve su contenido como un DataFrame de pandas.

    Args:
        filename (str): Nombre del archivo CSV a leer.

    Returns:
        pd.DataFrame: DataFrame con el contenido del archivo CSV.
    """
    file_path = os.path.join(INPUT_DIRECTORY, filename)
    df = pd.read_csv(file_path)
    print(f"Columnas en el CSV: {df.columns.tolist()}")
    return df


def write_excel(data: list, filename: str) -> None:
    file_path = os.path.join(OUTPUT_DIRECTORY, filename)
    
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        # Escribir datos originales (SplitterAgent)
        rows_original = []
        for item in data:
            original_message = item.get('original_message', '')
            message_type = item.get('message_type', '')
            categories = item.get('categories', [])
            for category in categories:
                keyword = category.get('keyword', '')
                for content in category.get('content', []):
                    rows_original.append({
                        'Original Message': original_message,
                        'Message Type': message_type,
                        'Category': keyword,
                        'Content': content
                    })
        
        if rows_original:
            df_original = pd.DataFrame(rows_original)
            df_original.to_excel(writer, index=False, sheet_name='Extracted Content')
            
            worksheet = writer.sheets['Extracted Content']
            for idx, col in enumerate(df_original.columns):
                max_length = max(df_original[col].astype(str).map(len).max(), len(col))
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 100)

        # Escribir datos de evaluación (EvaluatorAgent)
        if all(key in data[0] for key in ['precisión_del_tipo_de_mensaje', 'relevancia_de_la_categoría', 
                                          'exhaustividad_del_contenido', 'calidad_de_la_extracción', 
                                          'comentario', 'mensaje_original', 'puntuación_media']):
            df_eval = pd.DataFrame(data)
            columns_order = [
                'mensaje_original',
                'precisión_del_tipo_de_mensaje',
                'relevancia_de_la_categoría',
                'exhaustividad_del_contenido',
                'calidad_de_la_extracción',
                'puntuación_media',
                'comentario'
            ]
            df_eval = df_eval[columns_order]
            df_eval.to_excel(writer, index=False, sheet_name='Evaluación')
            
            worksheet = writer.sheets['Evaluación']
            for idx, col in enumerate(df_eval.columns):
                max_length = max(df_eval[col].astype(str).map(len).max(), len(col))
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 100)

    print(f"Resultados escritos en {file_path}")
    print(f"Total de filas escritas: Contenido Extraído - {len(rows_original)}, Evaluación - {len(data)}")