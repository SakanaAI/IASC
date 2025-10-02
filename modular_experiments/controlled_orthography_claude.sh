#!/bin/bash
## Create orthographies for the language(s) based on a given list of scripts.
## By default only does this for the first directory among the final
## morphosyntactic output directories, since this only depends on the phonology.
MODEL=claude
OUTPUT_DIR=modular_experiment_outputs_controlled
MORPHOSYNTAX_LANGUAGE=turkish
python3 modular_experiments/run_orthography.py \
        --model="${MODEL}" \
        --languages=Japanese \
        --morphosyntax_language="${MORPHOSYNTAX_LANGUAGE}" \
        --scripts=Cyrillic \
        --modular_experiment_outputs="${OUTPUT_DIR}" \
        --run_commands
