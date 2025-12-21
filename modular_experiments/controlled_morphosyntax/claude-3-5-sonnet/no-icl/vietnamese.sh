#!/bin/bash

MODEL=claude-3-5-sonnet
OUTPUT_DIR=modular_experiment_outputs_controlled
python3 modular_experiments/run_morphosyntax.py \
	--model="${MODEL}" \
	--modular_experiment_outputs="${OUTPUT_DIR}" \
	--story=grammatical_test_sentences \
	--storydir=sentence_design_output \
	--use_safe_params \
	--num_iter=1 \
	--premade_params_language="vietnamese" \
	--reference_file="evaluation/data/vietnamese.csv" 

. ./modular_experiment_outputs_controlled/claude-3-5-sonnet/vietnamese/metascripts/word_order.sh
. ./modular_experiment_outputs_controlled/claude-3-5-sonnet/vietnamese/metascripts/morphosyntax_0_0_0.sh

. ./modular_experiment_outputs_controlled/claude-3-5-sonnet/vietnamese/metascripts/evaluation_0_0.sh
