# Informal Register and LLMs

## Overview

This repository contains the data, annotations, and evaluation results from a study on how large language models (LLMs) handle **informal and colloquial language** in machine translation. Despite the impressive overall translation quality modern LLMs achieve, their ability to preserve informal register across languages remains largely unexplored — this project addresses that gap.

Two original corpora were compiled in **Catalan** and **Spanish**, systematically categorised according to a set of informal linguistic features. These corpora were then translated by three LLMs across two language pairs, and the resulting translations were subjected to human evaluation assessing both **register preservation** and **translation strategy adoption**.

---

## Language Pairs

| Source | Target |
|--------|--------|
| Catalan (CA) | Spanish (ES) |
| Spanish (ES) | English (EN) |

---

## Models Evaluated

| Model | Developer | Parameters |
|-------|-----------|------------|
| [Salamandra Instruct](https://huggingface.co/BSC-LT/salamandra-7b-instruct) | BSC Language Technologies | 7B |
| [TranslateGemma](https://huggingface.co/) | Google | 12B |
| [Qwen](https://huggingface.co/Qwen) | Alibaba Cloud | 8B |

---

## Repository Structure

```
.
├── datasets/
│   ├── catalan/          # Original Catalan corpus with informal feature annotations
│   └── spanish/          # Original Spanish corpus with informal feature annotations
├── translations/
│   ├── ca-es/            # Catalan → Spanish translations (all 3 models)
│   └── es-en/            # Spanish → English translations (all 3 models)
├── evaluation/
│   ├── human_eval/       # Human evaluation scores and annotations
├── annotation_schema/    # Informal feature taxonomy and annotation guidelines
└── README.md
```

---

## Informal Linguistic Features

The corpora were annotated according to a systematic taxonomy of informal features, including (but not limited to):

- **Colloquial vocabulary** — slang, everyday informal lexicon
- **Discourse markers** — fillers, hedges, informal connectors
- **Non-standard morphosyntax** — ellipsis, unconventional agreement patterns
- **Pragmatic markers** — irony, intensifiers, expressive particles
- **Register-specific address forms** — second-person variation (e.g. *tu/vos/usted* in Spanish)

Full annotation guidelines are available in `annotation_schema/`.

---

## Evaluation Methodology

Human evaluators assessed each translation along two dimensions:

1. **Register preservation** — Does the translation maintain the informal/colloquial tone of the source?
2. **Translation strategy** — Which strategy was adopted (e.g. domestication, foreignisation, neutralisation, compensation)?

---

## Key Findings

> Detailed results are available in `evaluation/results/`.

The study sheds light on how informal and colloquial language is handled — and often lost — in automated translation systems, contributing a meaningful addition to an underexplored area of machine translation research.

---

## Citation

If you use this dataset or findings in your work, please cite:

```bibtex
@misc{informal-register-mt,
  title     = {Informal Register in Machine Translation: A Corpus-Based Evaluation of LLMs},
  year      = {2025},
  note      = {Corpora and evaluation data for Catalan–Spanish and Spanish–English MT}
}
```

---

## License

This work is licensed under a **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License**.

[![CC BY-NC-ND 4.0](https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

**You are free to:**
- Share — copy and redistribute the material in any medium or format

**Under the following terms:**
- **Attribution (BY)** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial (NC)** — You may not use the material for commercial purposes.
- **NoDerivatives (ND)** — If you remix, transform, or build upon the material, you may not distribute the modified material.

Full license text: [https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode)

---

## Contact

For questions about the corpora, annotation schema, or evaluation methodology, please open an issue in this repository.

