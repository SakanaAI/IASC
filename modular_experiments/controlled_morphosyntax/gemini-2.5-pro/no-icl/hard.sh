#!/bin/bash

MODEL=gemini-2.5-pro
OUTPUT_DIR=modular_experiment_outputs_controlled
python3 modular_experiments/run_morphosyntax.py \
	--model="${MODEL}" \
	--modular_experiment_outputs="${OUTPUT_DIR}" \
	--story=grammatical_test_sentences \
	--storydir=sentence_design_output \
	--use_safe_params \
	--num_iter=1 \
	--premade_params_language="hard" \
	--reference_file="evaluation/data/hard.csv" 

. ./modular_experiment_outputs_controlled/gemini-2.5-pro/hard/metascripts/word_order.sh
. ./modular_experiment_outputs_controlled/gemini-2.5-pro/hard/metascripts/morphosyntax_0_0_0.sh

. ./modular_experiment_outputs_controlled/gemini-2.5-pro/hard/metascripts/evaluation_0_0.sh
