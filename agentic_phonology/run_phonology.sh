#!/bin/bash
while getopts "t:" flag;
do
    case $flag in
	t)
	    LANGUAGE="${OPTARG}"
	    ;;
	\?)
	    echo Invalid option
	    ;;
    esac
done
if [ -z "${LANGUAGE}" ]
then
    echo You must provide a non-empty language name to the -l flag.
    exit 1
fi
MODEL=claude
OUTDIR="agentic_phonology/outputs/${LANGUAGE}"
PHONOTACTICS=$(ls ${OUTDIR}/phonotactics_${MODEL}_??.py |tail -1)
echo PHONOTACTICS is "${PHONOTACTICS}"
python3 agentic_phonology/run_phonology_main.py \
	--which_task=phonrules \
	--model="${MODEL}" \
	--phonotactics="${PHONOTACTICS}" \
	--phonrules_base="${OUTDIR}/phonology_${MODEL}" \
	--max_iter=10 \
	--num_closest=1 \
	--num_output_examples=20 \
	--user_prompt_dump
