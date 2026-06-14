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
    source = "Spanish"
    target = "English"

    input_file  = Path("/home/lbuigues001/TFM/salamandraTA/dataset_ES.csv").resolve()
    output_file = Path("/home/lbuigues001/TFM/salamandraTA/output_ES_fewshot1.jsonl").resolve()

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

            text = f"""Translate the following extract of informal text from Spanish to English. The text comes from informal interviews published on YouTube. Text before and after the extract is also added to help you better understand the context. Produce a natural translation and make sure it maintains the colloquial tone and register. In addition, examples of translations are provided below to guide you:

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