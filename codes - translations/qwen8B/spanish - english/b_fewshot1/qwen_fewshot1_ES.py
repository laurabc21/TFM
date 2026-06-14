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

    input_file: Path = Path("/home/lbuigues001/TFM/qwen/dataset_ES.csv").resolve()
    output_file: Path = Path("/home/lbuigues001/TFM/qwen/output_fewshot1_ES.jsonl").resolve()

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

            prompt: str = f"""Translate the following extract of informal text from Spanish to English. The text comes from informal interviews published on YouTube. Text before and after the extract is also added to help you better understand the context. Produce a natural translation and make sure it maintains the colloquial tone and register. In addition, examples of translations are provided below to guide you:

Example 1:
ES: Buf, me costó un ojo de la cara, pero al final conseguí las entradas para el concierto.
EN: Phew, it cost me an arm and a leg, but I finally got the tickets for the concert.

Example 2:
ES: Chaval, ¿pa qué te metes en líos? Luego no vengas llorando.
EN: Dude, why d'you get yourself into trouble? Don't start crying later.

Example 3:
ES: Vaya, qué sorpresa, otra vez has metido la pata hasta el fondo. Eres un crack.
EN: Wow, what a surprise, you've really screwed up again. You're such a genius.

Example 4:
ES: La peli de Marvel fue una pasada, literal me quedé sin palabras, bro.
EN: The Marvel movie was actually insane, it literally left me speechless, bro.

Example 5:
ES: Claro, porque hacer ghosting a alguien es súper maduro, ¿no? O sea, qué nivel.
EN: Sure, because ghosting someone is super mature, right? I mean, impressive!

Now translate the following text using the information provided.

Previous text: {row.text_previous}
Text to be translated: {row.text_analyzed}
Following text: {row.text_after}

Translation:"""

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