#Informal Language and LLMs in Machine Translation
This repository contains the datasets, translations, and evaluation materials for a Master's dissertation examining how Large Language Models handle informal register in machine translation across Catalan–Spanish and Spanish–English language pairs.
Informal language constitutes a significant gap in both linguistic research and machine translation. Despite the remarkable progress LLMs have demonstrated in overall translation quality, their capacity to preserve informal register across languages remains largely unexplored. To address this gap, two original corpora in Catalan and Spanish were compiled and systematically categorised according to a set of informal linguistic features, then translated by three LLMs across two language pairs. The resulting translations were subjected to human evaluation assessing both register preservation and translation strategy adoption.

#Languages
LanguageRoleCatalanSource (CAT–ES)SpanishSource (ES–EN) / Target (CAT–ES)EnglishTarget (ES–EN)

#Models Evaluated
ModelSizeDeveloperSalamandra7BBarcelona Supercomputing Center (BSC)Qwen8BAlibaba CloudTranslateLLM (translategemma)12BGoogle

Datasets
Two corpora were compiled from publicly available YouTube content and annotated across four macro-categories of informal language:
CategoryCAT–ESES–EN1. Lexicon and Vocabulary1581502. Non-standard Linguistic Features7123. Pragmatics and Discourse28324. Register and Tone79Total200203

Repository Structure
├── data/               # Annotated source corpora (CAT–ES and ES–EN)
├── translations/       # LLM-generated translations
├── evaluation/         # Human evaluation scores and annotations
└── notebooks/          # Analysis and results

Citation
If you use this dataset or findings in your research, please cite accordingly.

License
CC BY-NC-ND 4.0 — See LICENSE.md
