#!/bin/bash
python3 transductions.py --output=diachronica_expanded.jsonl  --print_first_prompt
python3 transductions.py --previous_output=diachronica_expanded.jsonl --output=diachronica_expanded_2.jsonl  --print_first_prompt
