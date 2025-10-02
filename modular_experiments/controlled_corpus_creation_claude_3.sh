#!/bin/bash
## This assumes one has run the script in controlled_translation_claude.sh with
## the two sets of stories story_the_two_towers and
## story_a_clear_day_in_spring. It also assumes of course that the orthography
## has been created (e.g. controlled_orthography_claude.sh) and that the two
## corpus creation phases --- controlled_corpus_creation_claude_0.sh and
## controlled_corpus_creation_claude_1.sh have been run.
##
## It will then update the lexicon with the new
## words from the stories. Calls corpus/construct_corpus.py directly, since this
## is simpler in this case than going via the run_corpus_creation.py interface.
MODEL=claude
MORPHOSYNTAX_LANGUAGE=arabic
STORIES=sentence_design_output/grammatical_test_sentences.txt
STORIES="${STORIES},llm_stories/story_the_two_towers.txt"
STORIES="${STORIES},llm_stories/story_a_clear_day_in_spring.txt"
BASE="modular_experiment_outputs_controlled/${MODEL}"
MDIR="${BASE}/arabic/relativization_0_0_0/"
TRANS="${MDIR}/sentence_design_output.txt"
TRANS="${TRANS},${MDIR}/story_the_two_towers_translation.txt"
TRANS="${TRANS},${MDIR}/story_a_clear_day_in_spring_translation.txt"
PHONOTACTICS="${BASE}/phonology/Japanese/phonotactics_09.py"
OUTDIR="${MDIR}/Japanese/Latin"
ORTHOGRAPHY="${OUTDIR}/orthography.py"
python3 corpus/construct_corpus.py \
        --outdir="${OUTDIR}" \
        --sources="${STORIES}" \
        --glosses="${TRANS}" \
        --phonotactics="${PHONOTACTICS}" \
        --orthography="${ORTHOGRAPHY}" \
        --lexicon="${OUTDIR}/lexicon_orth.tsv"
