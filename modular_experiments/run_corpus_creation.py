"""Corpus creation.

Constructions the lexicons and aligned corpus JSONL files.
"""

import sys
import os

sys.path.append(os.path.abspath("."))

import copy
import glob

from absl import app
from absl import flags
from absl import logging
from jinja2 import Template
from llm import llm
from run_morphosyntax import STAGES
from utils.common_utils import find_last_stage, SCRIPT, TRANSLATION

# from modular_experiments.morphosyntax_params import PARAMS

LANGUAGES = flags.DEFINE_list(
    "languages",
    ["French", "Hawaiian", "Japanese", "Spanish", "Welsh"],
    "List of languages upon which the phonotactics were based",
)
MODULAR_EXPERIMENT_OUTPUTS = flags.DEFINE_string(
    "modular_experiment_outputs",
    "modular_experiment_outputs",
    "Subdirectory for the experimental data",
)
STORY = flags.DEFINE_string(
    "story",
    "story_the_two_towers",
    "Name of the story",
)
STORYDIR = flags.DEFINE_string(
    "storydir",
    "llm_stories",
    "Path to directory of stories.",
)
USE_ORTHOGRAPHY = flags.DEFINE_bool(
    "use_orthography",
    False,
    "The second stage, where we update with the orthography.",
)
MORPHOSYNTAX_LANGUAGE = flags.DEFINE_string(
    "morphosyntax_language",
    "turkish",
    "Language for morphosyntax (e.g., Turkish).",
)
RUN_COMMANDS = flags.DEFINE_bool(
    "run_commands",
    False,
    "Actually run the created commands.",
)
RUN_SINGLE = flags.DEFINE_bool(
    "run_single",
    False,
    "Run only a single experiment (index 0 iter).",
)

MODEL = llm.MODEL
OPEN_AI_API_KEY = llm.OPEN_AI_API_KEY


CORPUS_CMD = Template(
    """
python3 corpus/construct_corpus.py \\
  --outdir="{{outdir}}" \\
  --sources="{{story}}" \\
  --glosses="{{translation}}" \\
  --phonotactics="{{phonotactics}}"
""".strip()
)

ORTHOGRAPHY_ARGS = Template("""  --orthography="{{orthography}}" \\
  --script_outdir="{{script_outdir}}" \\
  --lexicon="{{lexicon}}"
""")


def find_latest(file_pattern):
    try:
        return max(glob.glob(file_pattern), key=os.path.getmtime)
    except ValueError:
        return file_pattern


def run_corpus_creation(language: str, dir: str) -> None:
    """Run the corpus creation for a given phonotactics.

    Args:
      language: Language upon the phonotactics were based.
      dir: Directory for the output of the grammar developer.
    """
    story = os.path.join(STORYDIR.value, f"{STORY.value}.txt")
    translation = os.path.join(dir, f"{TRANSLATION.value}.txt")

    phonotactics = find_latest(
        os.path.join(
            MODULAR_EXPERIMENT_OUTPUTS.value,
            llm.MODEL.value,
            "phonology",
            language,
            "phonotactics_*.py",
        ),
    )
    outdir = os.path.join(dir, language)
    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass
    cmd = CORPUS_CMD.render(
        outdir=outdir,
        story=story,
        translation=translation,
        phonotactics=phonotactics,
    )
    if USE_ORTHOGRAPHY.value:
        script_outdir = os.path.join(outdir, SCRIPT.value)
        orthography = find_latest(os.path.join(script_outdir, "orthography*.py"))
        if orthography.endswith("orthography*.py"):
            logging.warning(
                "No orthography*.py in "
                f"{script_outdir} or {script_outdir} does not exist"
            )
            return
        lexicon = os.path.join(outdir, "lexicon.tsv")
        cmd = " \\\n".join(
            [
                cmd,
                ORTHOGRAPHY_ARGS.render(
                    orthography=orthography,
                    script_outdir=script_outdir,
                    lexicon=lexicon,
                ),
            ],
        ).strip()
    print(cmd)
    if RUN_COMMANDS.value:
        failure = os.system(cmd)
        if failure:
            with open(os.path.join(outdir, "fail.txt"), "w") as stream:
                stream.write("Failed\n")


def main(unused_argv):
    if USE_ORTHOGRAPHY.value:
        assert SCRIPT.value is not None
    for language in LANGUAGES.value:
        output_files = glob.glob(
            os.path.join(
                MODULAR_EXPERIMENT_OUTPUTS.value,
                llm.MODEL.value,
                MORPHOSYNTAX_LANGUAGE.value, "*"
                )
            )

        last_stage = find_last_stage(stages=STAGES,
                                     output_files=output_files)
        if last_stage is None:
            raise ValueError("No stages found in the scripts directory.")
        print(f"Last stage found: {last_stage}")

        if RUN_SINGLE.value:
            dir = os.path.join(
                MODULAR_EXPERIMENT_OUTPUTS.value,
                llm.MODEL.value,
                MORPHOSYNTAX_LANGUAGE.value,
                f"{last_stage}_0_0_0",
            )
            run_corpus_creation(language, dir)
            return
        else:
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
                run_corpus_creation(language, dir)


if __name__ == "__main__":
    app.run(main)
