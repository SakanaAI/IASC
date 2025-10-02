#!/bin/bash
## Create phonologies (phonotactics) for the language based on a given list of
## languages.
MODEL=claude
OUTPUT_DIR=modular_experiment_outputs_controlled
python3 modular_experiments/run_phonology.py \
        --model="${MODEL}" \
        --languages="Japanese" \
        --modular_experiment_outputs="${OUTPUT_DIR}" \
        --run_commands
