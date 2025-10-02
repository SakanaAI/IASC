#!/bin/bash
## Creates a corpus of texts given the grammar, phonology and orthography.
##
## This assumes that only the first output language has an orthography.
MODEL=claude
OUTPUT_DIR=modular_experiment_outputs_controlled
MORPHOSYNTAX_LANGUAGE=turkish
python3 modular_experiments/run_corpus_creation.py \
        --model="${MODEL}" \
        --languages=Japanese \
        --use_orthography \
        --script=Cyrillic \
        --storydir=sentence_design_output \
        --story=grammatical_test_sentences \
        --translation=sentence_design_output \
        --morphosyntax_language="${MORPHOSYNTAX_LANGUAGE}" \
        --modular_experiment_outputs="${OUTPUT_DIR}" \
        --run_commands
