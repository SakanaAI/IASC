"""Runner for orthography series.

Creates commands to iteratively develop an orthographic model.
"""

import sys
import os

sys.path.append(os.path.abspath("."))

import glob

from absl import app
from absl import flags
from jinja2 import Template
from llm import llm

# from modular_experiments.morphosyntax_params import PARAMS
from typing import Any, Dict, Tuple

from run_morphosyntax import STAGES
from utils.common_utils import find_last_stage

LANGUAGES = flags.DEFINE_list(
    "languages",
    ["French", "Hawaiian", "Japanese", "Spanish", "Welsh"],
    "List of languages upon which the phonotactics were based",
)
MORPHOSYNTAX_LANGUAGE = flags.DEFINE_string(
    "morphosyntax_language",
    "turkish",
    "Language used in the morphosyntax experiment.",
)
SCRIPTS = flags.DEFINE_list(
    "scripts",
    ["Latin", "Cyrillic", "Arabic", "Greek"],
    "List of scripts to base the orthography on.",
)
MODULAR_EXPERIMENT_OUTPUTS = flags.DEFINE_string(
    "modular_experiment_outputs",
    "modular_experiment_outputs",
    "Subdirectory for the experimental data",
)
ONE_GRAMMAR_OUTPUT_ONLY = flags.DEFINE_bool(
    "one_grammar_output_only",
    True,
    "Run only for the first grammar directory in the glob.",
)
RUN_COMMANDS = flags.DEFINE_bool(
    "run_commands",
    False,
    "Actually run the created commands.",
)

MODEL = llm.MODEL
OPEN_AI_API_KEY = llm.OPEN_AI_API_KEY

COMMON_ARGS = Template("""   --model="{{model}}" \\
   --task="orthography" \\
   --user_prompt="prompts/orthography.txt" \\
   --open_ai_api_key={{open_ai_api_key}} \\
   --num_iterations=1""")

CREATE = "python3 create/create.py"


def create_common_args() -> str:
    """Creates the common set of arguments used by the experiments.

    Returns:
      Argument string.
    """
    common_args = COMMON_ARGS.render(
        open_ai_api_key=OPEN_AI_API_KEY.value or '""',
        model=MODEL.value,
    )
    return common_args


def find_latest(file_pattern):
    try:
        return max(glob.glob(file_pattern), key=os.path.getmtime)
    except ValueError:
        return file_pattern


def run_orthography_experiment(
    language: str,
    script: str,
    dir: str,
) -> None:
    """Run the orthography for a given language and

    Args:
      language: Language upon which to base the phonotactics.
      script: Script to use.
      dir: Directory for the output of the grammar developer.
    """
    # outdir = os.path.join(dir, language)
    outdir = os.path.join(dir, language) # TODO(ct): debug
    phonotactics = find_latest(
        os.path.join(
            MODULAR_EXPERIMENT_OUTPUTS.value,
            llm.MODEL.value,
            "phonology",
            language,
            "phonotactics_*.py",
        ),
    )
    script_outdir = os.path.join(outdir, script)
    try:
        os.makedirs(script_outdir)
    except FileExistsError:
        pass
    common_args = create_common_args()
    output = os.path.join(script_outdir, "orthography.py")
    output_full = os.path.join(script_outdir, "orthography_full.txt")
    user_prompt_dump = f"{script_outdir}/user_prompt_orthography.txt"
    cmd = " \\\n".join(
        [
            CREATE,
            common_args,
            f'   --output="{output}"',
            f'   --output_full="{output_full}"',
            f'   --lexicon="{outdir}/lexicon.tsv"',
            f'   --input_phonotactics="{phonotactics}"',
            f'   --language="{language}"',
            f'   --script="{script}"',
            f'   --user_prompt_dump="{user_prompt_dump}"',
        ]
    )
    print(cmd)
    if RUN_COMMANDS.value:
        failure = os.system(cmd)
        if failure:
            with open(os.path.join(script_outdir, "fail.txt"), "w") as stream:
                stream.write("Failed\n")


def main(unused_argv):
    for language in LANGUAGES.value:
        for script in SCRIPTS.value:
            output_files = glob.glob(
                os.path.join(
                    MODULAR_EXPERIMENT_OUTPUTS.value,
                    llm.MODEL.value,
                    MORPHOSYNTAX_LANGUAGE.value,
                    "*",
                )
            )
            if not output_files:
                print(
                    f"No grammar output files found for {language} in "
                    f"{os.path.join(MODULAR_EXPERIMENT_OUTPUTS.value, llm.MODEL.value, language, '*')}"
                )
                continue

            last_stage = find_last_stage(stages=STAGES,
                                         output_files=output_files)

            for dir in sorted(
                glob.glob(
                    os.path.join(
                        MODULAR_EXPERIMENT_OUTPUTS.value,
                        llm.MODEL.value,
                        MORPHOSYNTAX_LANGUAGE.value,
                        f"{last_stage}_?_?_?",
                    ),
                ),
            ):
                run_orthography_experiment(language=language,
                                           script=script,
                                           dir=dir)
                if ONE_GRAMMAR_OUTPUT_ONLY.value:
                    break


if __name__ == "__main__":
    app.run(main)
