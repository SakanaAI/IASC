#!/bin/bash
# Proof of concept for `translation` for one grammatical setting using Claude as
# the LLM.
MODEL=claude
OUTPUT_DIR=modular_experiment_outputs_controlled
STORIES=llm_stories
# Choose Arabic morphosyntactic settings since it has a DUAL. Find the final
# directory. This assumes you have run the phonology, morphosyntax, corpus
# creation and handbook creation scripts.
# Two examples one might use to illustrate the translation of DUAL:
STORY=story_the_two_towers
STORY=story_a_clear_day_in_spring
ODIR=$(ls -td modular_experiment_outputs_controlled/claude/arabic/* | sed 1q)
SAMPLE_TEXT=sentence_design_output/grammatical_test_sentences.txt
TRANSLATION="${ODIR}/sentence_design_output.txt"
# Choose Japanese phonology and Latin Orthography
HANDBOOK="${ODIR}/Japanese/Latin/handbook.txt"
echo "Base translation is ${TRANSLATION}"
echo "Base handbook is ${HANDBOOK}"
echo "Translating story: ${STORIES}/${STORY}.txt"
python3 create/create.py \
        --model="${MODEL}" \
        --task="new_translation" \
        --user_prompt="prompts/new_translation.txt" \
        --sample_text="${SAMPLE_TEXT}" \
        --translation="${TRANSLATION}" \
        --handbook="${HANDBOOK}" \
        --new_sample_text="${STORIES}/${STORY}.txt" \
        --output="${ODIR}/${STORY}_translation.txt" \
        --output_full="${ODIR}/${STORY}_translation_full.txt" \
        --open_ai_api_key="${OPEN_AI_API_KEY}" \
        --num_iterations=1
