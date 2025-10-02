"""Evaluation code for the output.

First, the output will be produced as a raw text file, often
containing the reasoning and other information.
For this reason, we need to structuralize the output.

Make sure that the OpenAI API key is set in the environment variables.
In my case, it is set as an environment variable in conda.
To see the list of env vars, run: `conda env config vars list`
To set the OpenAI API key, run:
`conda env config vars set OPENAI_API_KEY=<your_api_key_here>`
"""

import openai
from pydantic import BaseModel
import json
from typing import List, Any
import argparse


DEFAULT_SYSTEM_PROMPT = "You are a helpful assisntant."


def get_args() -> argparse.Namespace:
    """Get command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the output of the model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="Model to use for evaluation.",
    )
    parser.add_argument(
        "--user_prompt_template_file",
        type=str,
        default="eval_user_prompt_template.txt",
        help="Path to the user prompt template file.",
    )
    parser.add_argument(
        "--source_sentences_file",
        type=str,
        default="../sentence_design_output/grammatical_test_sentences.txt",
        help="Path to the source sentences file.",
    )
    parser.add_argument(
        "--translation_glosses_file",
        type=str,
        default="../modular_experiment_outputs_controlled/claude/extras_0_1_0/sentence_design_output.txt",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="translation_results.csv",
        help="Path to the output file to save the structuralized output.",
    )
    parser.add_argument(
        "--no_print",
        action="store_true",
        help="If set, do not print the output to the console.",
    )
    return parser.parse_args()


class SentencePair(BaseModel):
    """Structure for a sentence pair."""
    source_sentence: str
    translation_gloss: str


class StructuralizedOutput(BaseModel):
    """Structure for the model outputs."""
    sentence_pair_list: List[SentencePair]


def structuralize(model: str,
                  client: openai.OpenAI,
                  user_prompt: str,
                  system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                  text_format: BaseModel = StructuralizedOutput,
                  temperature: float = 0.0) -> List[Any]:
    """Structuralize the output text into a dictionary."""
    if not user_prompt:
        raise ValueError("User prompt template is required.")
    
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        text_format=text_format,
        temperature=temperature
    )
    # response.output_parsed (StructuralizedOutput)
    return response.output_parsed.sentence_pair_list


def main(args: argparse.Namespace) -> None:
    """Main evaluation function."""
    # some tests
    assert args.results_file.endswith(".csv"), "Output file must be a .csv file"
    
    client = openai.OpenAI()
    with open(args.user_prompt_template_file, "r") as f:
        user_prompt_template = f.read().strip()
        # this contains two placeholders: `source_sentences` and `translation_glosses`.
    
    # load source sentences
    with open(args.source_sentences_file, "r") as f:
        source_sentences = f.read().strip()
        # -> str
        
    # load translation glosses
    with open(args.translation_glosses_file, "r") as f:
        translation_glosses = f.read().strip()
        # -> str
    
    # format the user prompt
    input_text = f"### Source sentences\n{source_sentences}\n\n### Translation glosses\n{translation_glosses}"

    if not user_prompt_template.endswith("\n"):
        user_prompt_template += "\n"
        
    user_prompt = user_prompt_template + input_text
    
    # structuralize the output
    print(f"Structuralizing the output with {args.model}...")
    sentence_pair_list = structuralize(
        model=args.model,
        client=client,
        user_prompt=user_prompt,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        text_format=StructuralizedOutput,
        temperature=0.0
    )
    print(f"Structuralization completed. Found {len(sentence_pair_list)} sentence pairs.")
    
    if not args.no_print:
        for pair in sentence_pair_list:
            print(f"Source Sentence: {pair.source_sentence}")
            print(f"Translation Gloss: {pair.translation_gloss}")
            print("-" * 40)
    
    # save the structuralized output as a csv file
    with open(args.results_file, "w") as f:
        f.write("source_sentence,translation_gloss\n")
        for pair in sentence_pair_list:
            # pair is an instance of SentencePair (with keys `source_sentence` and `translation_gloss`)
            f.write(f"{pair.source_sentence},{pair.translation_gloss}\n")
    
    print(f"Structuralized output saved to {args.results_file}")
    

if __name__ == "__main__":
    args = get_args()
    main(args)