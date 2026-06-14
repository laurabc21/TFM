import transformers
import torch
from pathlib import Path
from typing import List, Dict, Union
import pandas as pd
import json
import os

if __name__ == "__main__":
    # Print current working directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    
    model_name: str = "google/translategemma-12b-it"
    
    # Use absolute path with resolve() to ensure it's correct
    input_file: Path = Path("/home/lbuigues001/TFM/translategemma/dataset_CAT.csv").resolve()
    output_file: Path = Path("/home/lbuigues001/TFM/translategemma/output_fewshot1_CAT.jsonl").resolve()
    
    source_lang_code: str = "ca-ES"
    target_lang_code: str = "es-ES"

    # Debug information
    print(f"Input file: {input_file}")
    print(f"File exists: {input_file.exists()}")
    print(f"Is file: {input_file.is_file()}")
    
    # Sanity check
    assert input_file.suffix.lower() == ".csv" and input_file.is_file(), f"File not found: {input_file}"
    assert output_file.suffix.lower() == ".jsonl" and output_file.parent.is_dir(), f"Output directory not found: {output_file.parent}"

    # Load the csv dataset
    df: pd.DataFrame = pd.read_csv(input_file, sep=";", encoding="latin-1")
    df = df.drop(columns=['previous_id', 'after_id', 'instance_analyzed_id'])
    
    print(f"Loaded {len(df)} rows from CSV")

    # Initialize the pipeline
    generator = transformers.pipeline(
        "image-text-to-text",
        model=model_name,
        device="cuda",
        dtype=torch.bfloat16,
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, row in enumerate(df.itertuples(), 1):
            print(f"Translating row {idx}/{len(df)}...", end='\r')
            
            prompt = f"""Traduce el siguiente extracto de texto informal del catalán al español. El texto proviene de podcasts y entrevistas informales publicadas en Youtube.
Se añade también texto anterior y posterior al extracto para ayudarte a comprender mejor el contexto. Produce una traducción natural y asegúrate de que mantenga el tono y registro coloquial.

Además, a continuación, se te proporcionan ejemplos de traducciones para guiarte:

Ejemplo 1 :
CA: O sigui aquest tio és un pringat de collons.
ES: O sea este tío es un pringado de cojones.

Ejemplo 2:
CAT: Doncs tio, a mi em van dir que venia a les set, però no sé, al final va arribar a les nou i mitja, el capullo.
ES: Pues tio, a mí me dijeron que venía a las siete, pero no sé, al final llegó a las nueve y media, el capullo.

Ejemplo 3:
CA : Buf tio, aquest examen era l'hòstia de difícil… Al final m'he quedat en blanc.
ES: Buf tío, este examen era la hostia de difícil… Al final me he quedado en blanco.

Ejemplo 4:
CA: Quin cringe, quasi em moro de la vergonya durant l'actuació.
ES: Qué cringe, casi me muero de la vergüenza durante la actuación.

Ejemplo 5: 
CA: Saps què? Que se’n vagi a fer punyetes perquè ja n’estic ben farta de treure-li les castanyes del foc. 
ES: ¿Sabes qué? Que se vaya a hacer puñetas porque ya estoy harta de sacarle las castañas del fuego.

Ahora traduce el siguiente texto utilizando la información proporcionada.

Texto anterior:
{row.text_previous}

Texto a traducir:
{row.text_analyzed}

Texto posterior:
{row.text_after}

Traducción:
"""

            message: List[Dict[str, Union[str, List[Dict[str, str]]]]] = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": source_lang_code,
                            "target_lang_code": target_lang_code,
                            "text": prompt,
                        }
                    ],
                }
            ]
            
            raw_output: str = generator(
                 text=message,
                 max_new_tokens=800,
                 do_sample=False,
            )[0]["generated_text"][-1]["content"]

            # ---- CLEAN OUTPUT (remove prompt if echoed) ----
            translation = raw_output.strip()

            if translation.startswith("Traducción:"):
                translation = translation[len("Traducción:"):].strip()

            f.write(json.dumps({
                "text_previous": row.text_previous,
                "text_analyzed": row.text_analyzed,
                "text_after": row.text_after,
                "translation": translation
            }, ensure_ascii=False) + '\n')
    
    print(f"\nCompleted! Translated {len(df)} rows to {output_file}")
