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
    if text.startswith("Translation:"):
        text = text[len("Translation:"):].strip()
    return text

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")

    model_name: str = "Qwen/Qwen3-8B"

    input_file: Path = Path("/home/lbuigues001/TFM/qwen/faltan_cot1.csv").resolve()
    output_file: Path = Path("/home/lbuigues001/TFM/qwen/faltan_CoT1_ES.jsonl").resolve()

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

            prompt: str = f"""Translate the following extract of informal text from Spanish to English. The text comes from informal interviews published on YouTube.
Text before and after the extract is also included to help you better understand the context. Think step by step before translating to ensure you maintain the colloquial tone and register.

Previous text:
{row.text_previous}

Text to be translated:
{row.text_analyzed}

Following text:
{row.text_after}

Translation:
"""

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