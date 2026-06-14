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
    output_file: Path = Path("/home/lbuigues001/TFM/translategemma/output_fewshot2_ES.jsonl").resolve()
    
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
            
            prompt = f"""Translate the following informal text excerpt from Spanish to English. The text comes from informal interviews published on YouTube.
Text before and after the excerpt is also included to help you better understand the context. Produce a natural translation and ensure it maintains the colloquial tone and register.
In addition, below you will find examples of translations to guide you:

Example 1: 
ES: Este tío es un imbécil, siempre llega tarde y ni si disculpa.
EN: This guy is a jerk, always showing up late and doesn’t even say sorry.

Example 2: 
ES: No me rayes con eso ahora que estoy ocupado.
EN: Don’t start with that now,  I’m busy.

Example 3: 
ES: Nos vemos en la uni sobre las 15, ¿vale? Así estudiaremos juntos.
EN: See you at uni at 15 about 3, okay? We can study together.

Example 4: 
ES: Me importa un pimiento lo que piense la gente, voy a hacer lo que quiera.
EN: I don't give a damn what people think,  I'm gonna do as I please.

Example 5: 
ES: Hace un frío que te cagas, no salgas sin abrigo.
EN: It’s goddamn cold, don't go out without a coat and wrap up!

Example 6: 
ES: Con esto de la inteligencia artificial me siento atrapado en un capítulo de Black Mirror.
EN: This AI stuff has me living in a Black Mirror episode.

Example 7:
ES: Me di un golpe en la cabeza, ¡pam!, y me quedé medio atontado.
EN:  I had a blow on my head, bam!, and I was kinda dazed.

Example 8: 
ES: Voy a hacerme un selfie con el perro para subirlo a Instagram.
EN: I'm gonna take a selfie with the dog to post it on Instagram.

Example 9: 
ES: Ná', déjalo, no te preocupes por eso ahora.
EN: Nah, forget it, don't worry about that now.

Example 10: 
ES: Es que no me apetece salir hoy, ¿vale? Déjame en paz.
EN: I just don't feel like going out today, okay? Leave me alone.

Example 11: 
ES: Bueno, tío, ya te lo contaré luego con más calma.
EN: Well, man, I'll tell you when I have some peace and quiet.

Example 12: 
ES: Bro, no te vas a creer lo que me pasó ayer.
EN: Bro, you'll never believe what happened to me yesterday.

Example 13: 
ES: Qué bien, otra reunión más, justo lo que necesitaba para alegrarme el día.
EN: Great, another meeting, just my day!

Example 14: 
ES: Hace un calor que te mueres, no se puede salir a la calle.
EN: It's hot as hell! you can't go into the street.

Now translate the following text using the information provided.

Previous text:
{row.text_previous}

Text to be translated:
{row.text_analyzed}

Later text:
{row.text_after}

Translation:

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
