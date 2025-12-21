#!/bin/bash

MODEL=gpt-5
OUTPUT_DIR=modular_experiment_outputs_controlled
python3 modular_experiments/run_morphosyntax.py \
	--model="${MODEL}" \
	--modular_experiment_outputs="${OUTPUT_DIR}" \
	--story=grammatical_test_sentences \
	--storydir=sentence_design_output \
	--use_safe_params \
	--num_iter=1 \
	--premade_params_language="mizo" \
	--reference_file="evaluation/data/mizo.csv"  \
    --do_review

. ./modular_experiment_outputs_controlled/gpt-5/mizo_icl/metascripts/word_order.sh
. ./modular_experiment_outputs_controlled/gpt-5/mizo_icl/metascripts/morphosyntax_0_0_0.sh

. ./modular_experiment_outputs_controlled/gpt-5/mizo_icl/metascripts/evaluation_0_0.sh
