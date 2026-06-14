from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
import pandas as pd
import json
import re

def clean_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^(Spanish|Español|Castellano|Traducción)\s*:\s*', '', text, flags=re.IGNORECASE)
    return text.strip()

if __name__ == "__main__":
    model_id = "BSC-LT/salamandra-7b-instruct"
    source = "Catalan"
    target = "Spanish"

    input_file  = Path("/home/lbuigues001/TFM/salamandraTA/dataset_CAT.csv").resolve()
    output_file = Path("/home/lbuigues001/TFM/salamandraTA/output_CAT_fewshot1.jsonl").resolve()

    assert input_file.is_file(), f"File not found: {input_file}"
    assert output_file.parent.is_dir(), f"Output directory not found: {output_file.parent}"

    df = pd.read_csv(input_file, sep=";", encoding="latin-1")
    df = df.drop(columns=['previous_id', 'after_id', 'instance_analyzed_id'], errors='ignore')
    print(f"Loaded {len(df)} rows from CSV")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, row in enumerate(df.itertuples(), 1):
            print(f"Translating row {idx}/{len(df)}...", end='\r')

            text = f"""Traduce el siguiente extracto de texto informal del catalán al español.
El texto proviene de podcasts y entrevistas informales publicadas en Youtube.
Se añade también texto anterior y posterior al extracto para ayudarte a comprender mejor el contexto.
Produce una traducción natural y asegúrate de que mantenga el tono y registro coloquial.

Ejemplos:

Ejemplo 1:
CA: O sigui aquest tio és un pringat de collons.
ES: O sea este tío es un pringado de cojones.

Ejemplo 2:
CA: Doncs tio, a mi em van dir que venia a les set, però no sé, al final va arribar a les nou i mitja, el capullo.
ES: Pues tío, a mí me dijeron que venía a las siete, pero no sé, al final llegó a las nueve y media, el capullo.

Ejemplo 3:
CA: Buf tio, aquest examen era l'hòstia de difícil… Al final m'he quedat en blanc.
ES: Buf tío, este examen era la hostia de difícil… Al final me he quedado en blanco.

Ejemplo 4:
CA: Quin cringe, quasi em moro de la vergonya durant l'actuació.
ES: Qué cringe, casi me muero de la vergüenza durante la actuación.

Ejemplo 5:
CA: Saps què? Que se'n vagi a fer punyetes perquè ja n'estic ben farta de treure-li les castanyes del foc.
ES: ¿Sabes qué? Que se vaya a hacer puñetas porque ya estoy harta de sacarle las castañas del fuego.

Ahora traduce el siguiente texto:

Texto anterior:
{row.text_previous}

Texto a traducir (IMPORTANTE):
{row.text_analyzed}

Texto posterior:
{row.text_after}

Traducción:"""

            message = [{"role": "user", "content": text}]
            prompt = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
            input_length = inputs.shape[1]

            outputs = model.generate(
                input_ids=inputs.to(model.device),
                max_new_tokens=2400,
                do_sample=True,
                temperature=1.0,
                top_p=1.0
            )

            raw_output = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
            translation = clean_output(raw_output)

            # Debug (optional but recommended)
            # print("\nINPUT:", row.text_analyzed)
            # print("OUTPUT:", raw_output)

            f.write(json.dumps({
                "text_previous": row.text_previous,
                "text_analyzed": row.text_analyzed,
                "text_after":    row.text_after,
                "translation":   translation,
            }, ensure_ascii=False) + '\n')

    print(f"\nCompleted! Translated {len(df)} rows to {output_file}")