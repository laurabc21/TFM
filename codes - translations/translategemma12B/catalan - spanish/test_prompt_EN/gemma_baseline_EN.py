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
    input_file: Path = Path("/home/lbuigues001/TFM/translategemma/dataset_ES.csv").resolve()
    output_file: Path = Path("/home/lbuigues001/TFM/translategemma/output_ES_promptEN.jsonl").resolve()
    
    source_lang_code: str = "es-ES"
    target_lang_code: str = "en-GB"

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
            
            prompt = f"""Translate the following extract of informal text from Catalan to Spanish. The text comes from informal podcasts and interviews posted on Youtube.
Text before and after the extract is also provided to help you better understand the context. Produce a natural translation and make sure it maintains the colloquial tone and register.

Previous context:
{row.text_previous}

Text to translate:
{row.text_analyzed}

Next context:
{row.text_after}

Translation:"""

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
