#!/bin/bash
## Writes the grammar handbook for the language.
##
## This assumes that only the first output language has an orthography and
## orthographically transcribed corpus.
MODEL=claude
OUTPUT_DIR=modular_experiment_outputs_controlled
python3 modular_experiments/run_handbook.py \
        --model="${MODEL}" \
        --languages=Japanese \
        --morphosyntax_language=turkish \
        --scripts=Cyrillic \
        --story=grammatical_test_sentences \
        --modular_experiment_outputs="${OUTPUT_DIR}" \
        --run_commands \
        --one_grammar_output_only
