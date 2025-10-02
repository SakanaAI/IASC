#!/bin/bash
## Writes a set of scripts to create the morphosyntax in
##
## $OUTPUT_DIR/$MODEL/metascripts
##
## Note that by default this does not actually run the scripts, just sets them up.
##
## After running this script first run the script named `word_order.sh` then you can
## run the other scripts named morphosyntax_?_?_?.sh (order doesn't matter).
##
## The num_iter flag sets how many times to run the word order module for each
## configuration. For each of these runs, there will be one run for each word
## order variant chosen in the parameters, and then for each of those, one run
## of each of the other morphosyntactic bundles.

MODEL="gemini-2.5-flash"
OUTPUT_DIR=modular_experiment_outputs_controlled
python3 modular_experiments/run_morphosyntax.py \
	--model="${MODEL}" \
	--modular_experiment_outputs="${OUTPUT_DIR}" \
	--story=grammatical_test_sentences \
	--storydir=sentence_design_output \
	--use_safe_params \
	--num_iter=3 \
	--premade_params_language="turkish" # you can change this to other languages like "french"
