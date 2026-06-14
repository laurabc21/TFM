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
    output_file: Path = Path("/home/lbuigues001/TFM/translategemma/output_fewshot2_CAT.jsonl").resolve()
    
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
Se añade también texto anterior y posterior al extracto para ayudarte a comprender mejor el contexto. Produce una traducción natural y asegúrate de que mantenga el tono y el registro coloquial.

Además, a continuación, se te proporcionan ejemplos de traducciones para guiarte:

Ejemplo 1:
CA: El meu coordinador és un capullo integral.
ES: Mi coordinador es un capullo integral.

Ejemplo 2: 
CA: La festa d'ahir va ser una passada. Hi havia molta penya i un ambient molt guai.
ES: La fiesta de ayer fue una pasada. Había mucha peña y un ambiente muy guay.

Ejemplo 3: 
CA: Profe, una pregunta, perquè no m'ha quedat clar això de l'examen de mates.
ES: Profe, una pregunta, porque no me ha quedado claro lo del examen de mates.

Ejemplo 4: 
CA: No fas ni un brot, Ramon, hauries de posar-te les piles amb els exàmens.
ES: No das palo al agua, Ramón, tendrías que ponerte las pilas con los exámenes.

Ejemplo 5:
CA: Estic supercansada. Ahir vaig dormir fatal i avui tinc una pila de feina.
ES: Estoy supercansada. Ayer dormí fatal y hoy tengo un montón de trabajo.

Ejemplo 6: 
CA: Només de veure els espaguetis amb quètxup tinc records del Vietnam del menjador de l’escola.
ES: Solo de ver espaguetis con kétchup tengo recuerdos de Vietnam del comedor del colegio.

Ejemplo 7:
CA: Anava pel carrer supertranquil, i de sobte, bum! Em vaig entropessar amb una pedra.
ES: Iba por la calle supertranquilo y, de repente, ¡!bum! Me tropecé con una piedra.

Ejemplo 8: 
CA: No pare de scrollejar a Instagram i només veig stories de la sèrie que estic mirant. Estic fart!
ES: No paro de scrollear en Instagram y solo veo stories de la serie que me estoy viendo. ¡Estoy harto!

Ejemplo 9: 
CA: Que si vaig a la festa? Doncs clar que sí! Que no em coneixes o què? Jo mai em perdo una bona festa.
ES: ¿Que si voy a la fiesta? ¡Pues claro que sí! ¿Qué no me conoces o qué? Yo nunca me pierdo una buena fiesta.

Ejemplo 10: 
CA: O sigui al final Martina no va vindre a casa perquè havia de fer molts deures.
ES: O sea al final Martina no vino a casa porque tenía que hacer muchos deberes.

Ejemplo 11:
CA: Amore, no sé si eres conscient, però, saps que Pablo t’està posant les banyes amb Gemma, oi?
ES: Amore, no sé si eres consciente, pero, sabes que Pablo te está poniendo los cuernos con Gemma, ¿no?

Ejemplo 12: 
CA: Ah sí, arribar dues hores tard sense avisar és de persona superfiable, oi? Quina cara, tio.
ES: Ah sí, llegar dos horas tarde sin avisar es de persona superfiable, ¿no? Qué cara, tío. 

Ejemplo 13: 
CA: Aquest examen m'ha costat la vida, t'ho juro. He estudiat tres mil hores i ara estic rebentada.
ES: Este examen me ha costado la vida, te lo juro. He estudiado tres mil horas y ahora estoy reventada.

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
