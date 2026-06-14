import transformers
import torch
from pathlib import Path
import pandas as pd
import json
import os
import re

def clean_output(text: str) -> str:
    # Remove <think>...</think> blocks if present (fallback safety net)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.strip()
    if text.startswith("Traducción:"):
        text = text[len("Traducción:"):].strip()
    return text

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")

    model_name: str = "Qwen/Qwen3-8B"

    input_file: Path = Path("/home/lbuigues001/TFM/qwen/faltan_fewhsot1.csv").resolve()
    output_file: Path = Path("/home/lbuigues001/TFM/qwen/output_faltan.jsonl").resolve()

    print(f"Input file: {input_file}")
    print(f"File exists: {input_file.exists()}")
    print(f"Is file: {input_file.is_file()}")

    assert input_file.suffix.lower() == ".csv" and input_file.is_file(), f"File not found: {input_file}"
    assert output_file.suffix.lower() == ".jsonl" and output_file.parent.is_dir(), f"Output directory not found: {output_file.parent}"

    df: pd.DataFrame = pd.read_csv(input_file, sep=";", encoding="latin-1")
    df = df.drop(columns=['previous_id', 'after_id', 'instance_analyzed_id'], errors='ignore')
    print(f"Loaded {len(df)} rows from CSV")

    generator = transformers.pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, row in enumerate(df.itertuples(), 1):
            print(f"Translating row {idx}/{len(df)}...", end='\r')

            prompt: str = f"""Traduce el siguiente extracto de texto informal del catalán al español. El texto proviene de podcasts y entrevistas informales publicadas en Youtube. Se añade también texto anterior y posterior al extracto para ayudarte a comprender mejor el contexto. Produce una traducción natural y asegúrate de que mantenga el tono y registro coloquial. 
            Además, a continuación, se te proporcionan ejemplos de traducciones para guiarte:

Ejemplo 1:
CA: O sigui aquest tio és un pringat de collons.
ES: O sea este tío es un pringado de cojones.

Ejemplo 2:
CAT: Doncs tio, a mi em van dir que venia a les set, però no sé, al final va arribar a les nou i mitja, el capullo.
ES: Pues tio, a mí me dijeron que venía a las siete, pero no sé, al final llegó a las nueve y media, el capullo.

Ejemplo 3:
CA: Buf tio, aquest examen era l'hòstia de difícil… Al final m'he quedat en blanc.
ES: Buf tío, este examen era la hostia de difícil… Al final me he quedado en blanco.

Ejemplo 4:
CA: Quin cringe, quasi em moro de la vergonya durant l'actuació.
ES: Qué cringe, casi me muero de la vergüenza durante la actuación.

Ejemplo 5:
CA: Saps què? Que se'n vagi a fer punyetes perquè ja n'estic ben farta de treure-li les castanyes del foc.
ES: ¿Sabes qué? Que se vaya a hacer puñetas porque ya estoy harta de sacarle las castañas del fuego.

Ahora traduce el siguiente texto utilizando la información proporcionada.

Texto anterior: {row.text_previous}
Texto a traducir: {row.text_analyzed}
Texto posterior: {row.text_after}

Traducción:"""

            # Wrap prompt in chat template with thinking disabled
            messages = [{"role": "user", "content": prompt}]
            text = generator.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # Disables Qwen3 internal reasoning
            )

            raw_output: str = generator(
                text,
               max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
            )[0]["generated_text"]

            translation = clean_output(raw_output)

            f.write(json.dumps({
                "text_previous": row.text_previous,
                "text_analyzed": row.text_analyzed,
                "text_after": row.text_after,
                "translation": translation,
            }, ensure_ascii=False) + '\n')

    print(f"\nCompleted! Translated {len(df)} rows to {output_file}")