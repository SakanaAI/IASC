"""Runner for phonology series.

Creates commands to iteratively develop a set of phonotactic models based on one
or more languages.

- Use `argparse` instead of `absl` for command line arguments. (for my personal preference)
- Change the indent style from 2 spaces to 4 spaces.
- Remove unused imports.
"""
import sys
import os
sys.path.append(os.path.abspath("."))

from jinja2 import Template # for creating command templates
import argparse


def get_args() -> argparse.Namespace:
    """Get the command line arguments."""
    parser = argparse.ArgumentParser(description="Run phonology experiments.")
    parser.add_argument(
        "-l",
        "--languages",
        type=str,
        choices=["French", "Hawaiian", "Japanese", "Spanish", "Welsh"],
        default="Japanese",
        help="List of languages upon which to base the phonotactics",
    )
    parser.add_argument(
        "-o",
        "--modular_experiment_outputs",
        type=str,
        default="modular_experiment_outputs",
        help="Subdirectory for the experimental data",
    )
    parser.add_argument(
        "-r",
        "--run_commands",
        action="store_true",
        help="Actually run the created commands.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="claude",
        help="Model to use for the experiments.",
    )
    parser.add_argument(
        "--open_ai_api_key",
        type=str,
        default=None,
        help="OpenAI API key to use for the experiments.",
    )
    parser.add_argument(
        "-L",
        "--legacy_phonotactics",
        action="store_true",
        help="Run legacy phonotactics.",
    )

    return parser.parse_args()


LEGACY_COMMON_ARGS = Template("""   --model="{{model}}" \\
   --task="phonotactics,phonotactics_improvement" \\
   --user_prompt="prompts/phonotactics.txt,prompts/phonotactics_improvement.txt" \\
   --open_ai_api_key={{open_ai_api_key}} \\
   --num_iterations=5"""
)

COMMON_ARGS = Template("""   --model="{{model}}" \\
   --which_task="phonotactics" \\
   --num_closest=1 \\
   --num_output_examples=20 \\
   --user_prompt_dump \\
   --max_iter=10 \\
   --open_ai_api_key={{open_ai_api_key}}"""
)

CREATE_LEGACY = "python3 create/create.py"
CREATE_AGENTIC = "python3 agentic_phonology/run_phonology_main.py"


def create_common_args(open_ai_api_key: str | None,
                       model: str,
                       legacy: bool) -> str:
    """Creates the common set of arguments used by the experiments.

    Returns:
        Argument string.
    """
    if legacy:
        return LEGACY_COMMON_ARGS.render(
            open_ai_api_key=open_ai_api_key if open_ai_api_key else '""',
            model=model,
        )
    else:
        return COMMON_ARGS.render(
            open_ai_api_key=open_ai_api_key if open_ai_api_key else '""',
            model=model,
        )



def create_output_dir(language: str,
                      modular_experiment_outputs: str,
                      model: str) -> str:
    """Creates the output directory for the experiment.

    Args:
        language: Language upon which to base the phonotactics.
    Returns:
        Path to created output directory.
    """
    base_dir = os.path.join(
        modular_experiment_outputs,
        model,
        "phonology",
    )
    path = os.path.join(base_dir, language)
    os.makedirs(path, exist_ok=True)
    return path


def run_phonology_experiment(language: str,
                             open_ai_api_key: str,
                             model: str,
                             modular_experiment_outputs: str,
                             run_commands: bool) -> None:
    """Run the phonology for a given language.

    Args:
        language: Language upon which to base the phonotactics.
    """
    output_dir = create_output_dir(language,
                                   modular_experiment_outputs,
                                   model)
    output = os.path.join(output_dir, "phonotactics.py")
    output_full = os.path.join(output_dir, "phonotactics_full.txt")
    user_prompt_dump = f"{output_dir}/user_prompt.txt"
    if args.legacy_phonotactics:
        create = CREATE_LEGACY
        cmd = " \\\n".join(
            [
                create,
                create_common_args(open_ai_api_key, model, True),
                f'   --output="{output}"',
                f'   --output_full="{output_full}"',
                f'   --language="{language}"',
                f'   --user_prompt_dump="{user_prompt_dump}"',
            ]
        )
    else:
        create = CREATE_AGENTIC
        # Agentic version creates files from base version.
        output = output.replace(".py", "")
        cmd = " \\\n".join(
            [
                create,
                create_common_args(open_ai_api_key, model, False),
                f'   --language="{language}"',
                f'   --phonotactics_base={output}',
            ]
    )
    print(cmd)
    if run_commands:
        os.system(cmd)


def main(args: argparse.Namespace) -> None:
    """Main function to run phonology experiments.
    Args:
        args: Command line arguments.
    """
    if isinstance(args.languages, str):
        languages = [args.languages]
    elif isinstance(args.languages, list):
        languages = args.languages
    else:
        raise ValueError("Invalid type for languages argument. Expected str or list.")

    for language in languages:
        run_phonology_experiment(language,
                                 args.open_ai_api_key,
                                 args.model,
                                 args.modular_experiment_outputs,
                                 args.run_commands)


if __name__ == "__main__":
    args = get_args()
    main(args)
