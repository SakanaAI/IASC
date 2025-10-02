#!/bin/bash
## Creates a corpus of texts given the grammar and phonology.
MODEL=claude
OUTPUT_DIR=modular_experiment_outputs_controlled
MORPHOSYNTAX_LANGUAGE=turkish
python3 modular_experiments/run_corpus_creation.py \
	--model="${MODEL}" \
	--languages=Japanese \
	--storydir=sentence_design_output \
	--story=grammatical_test_sentences \
	--translation=sentence_design_output \
	--modular_experiment_outputs="${OUTPUT_DIR}" \
	--morphosyntax_language="${MORPHOSYNTAX_LANGUAGE}" \
	--run_single \
	--run_commands
	