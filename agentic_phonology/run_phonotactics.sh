#!/bin/bash
while getopts "l:" flag;
do
    case $flag in
        l)
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
python3 agentic_phonology/run_phonology_main.py \
        --which_task=phonotactics \
        --model="${MODEL}" \
        --language="${LANGUAGE}" \
        --phonotactics_base="${OUTDIR}/phonotactics_${MODEL}" \
        --max_iter=10 \
        --num_closest=1 \
        --num_output_examples=20 \
        --user_prompt_dump
