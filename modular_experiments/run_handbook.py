"""Runner for handbook creation."""

import sys
import os

sys.path.append(os.path.abspath("."))

import collections

# import copy
import glob

from absl import app
from absl import flags
from jinja2 import Template
from llm import llm
from run_morphosyntax import STAGES
from morphosyntax_params import LANGUAGE_TO_PARAMS, Morphosyntax
from utils.common_utils import find_last_stage

# from modular_experiments.morphosyntax_params import PARAMS
from typing import Any, Dict, Tuple

LANGUAGES = flags.DEFINE_list(
    "languages",
    ["French", "Hawaiian", "Japanese", "Spanish", "Welsh"],
    "List of languages upon which the phonotactics were based",
)
SCRIPTS = flags.DEFINE_list(
    "scripts",
    ["Latin", "Cyrillic", "Arabic", "Greek", "Hangul-Jamo"],
    "List of scripts to base the orthography on.",
)
MORPHOSYNTAX_LANGUAGE = flags.DEFINE_string(
    "morphosyntax_language",
    "turkish",
    "Language used in the morphosyntax experiment.",
)
STORY = flags.DEFINE_string(
    "story",
    "story_the_two_towers",  # Good for both dual and incl/excl.
    "Name of the story",
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
   --task="handbook" \\
   --user_prompt="prompts/handbook.txt" \\
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


def load_params(path: str) -> Dict[str, str]:
    """Loads parameters from stored params text file."

    Args:
      path: Path to params.
    Returns:
      Dictionary
    """
    params = collections.defaultdict(str)
    with open(path) as stream:
        for line in stream:
            param, value = line.split(":")
            params[param.strip()] = value.strip()
    return params


def load_params_premade(morphosyntax_language: str) -> Morphosyntax:
    """Load pre-defined Morphosyntax parameters.

    Args:
        language: Language name. Make sure that the parameters
            for the language is already defined in morphosyntax_params.py.
    Returns:
        Dictionary of parameters.
    """
    params = LANGUAGE_TO_PARAMS.get(morphosyntax_language)
    if params is None:
        raise ValueError(f"No parameters defined for language: {morphosyntax_language}")
    return params()


def run_handbook_experiment(
    language: str,
    morphosyntax_language: str,
    script: str,
    dir: str,
) -> None:
    """Run the orthography for a given language and

    Args:
      language: Language upon which to base the phonotactics.
      script: Script to use.
      dir: Directory for the output of the grammar developer.
    """
    params = load_params_premade(morphosyntax_language)
    syntax = params.syntax
    morphology = params.morphology

    # syntax
    main_word_order = syntax.main_word_order or "NONE"
    adj_noun_word_order = syntax.adj_noun_word_order or "NONE"
    posspron_noun_word_order = syntax.posspron_noun_word_order or "NONE"
    num_noun_word_order = syntax.num_noun_word_order or "NONE"
    morphology_type = syntax.morphology_type or "NONE"
    adposition_noun_word_order = syntax.adposition_noun_word_order or "NONE"
    alignment = syntax.alignment or "NONE"

    # morphology
    pro_drop = morphology.pro_drop or "NONE" # currently not used (as of Sep 18, 2025)
    case = morphology.case or "NONE"
    gender = morphology.gender or "NONE" # currently not used (as of Sep 18, 2025)
    definiteness = morphology.definiteness or "NONE"
    adjective_agreement = morphology.adjective_agreement or "NONE"
    nominal_number = morphology.nominal_number or "NONE"
    comparative = morphology.comparative or "NONE"
    tense_aspect = morphology.tense_aspect or "NONE"
    person = morphology.person or "NONE"
    mood = morphology.mood or "NONE"
    voice = morphology.voice or "NONE"
    relativization = morphology.relativization or "NONE"
    infinitive = morphology.infinitive or "NONE"
    negation = morphology.negation or "NONE"
    inclusive_exclusive = morphology.inclusive_exclusive or "NONE"

    # params = load_params(os.path.join(dir, "params.txt"))

    # Old morphology settings:
    """"
    number_marking = params["nominal_number_marking"] or "NONE"
    case_marking = params["nominal_case_marking"] or "NONE"
    tense_aspect_marking = params["verbal_tense_aspect_marking"] or "NONE"
    person_agreement = params["verbal_person_agreement"] or "NONE"
    # This is not used. TODO(rws): eventually deprecate this.
    head_marking = "NONE"
    main_word_order = params["main_word_order"] or "NONE"
    adj_noun_word_order = params["adj_noun_word_order"] or "NONE"
    adposition_noun_word_order = params["adposition_noun_word_order"] or "NONE"
    gender_marking = params["gender_marking"] or "NONE"
    numeral_classifier = params["numeral_classifier"] or "NONE"
    inclusive_exclusive = params["inclusive_exclusive"] or "NONE"
    """

    outdir = os.path.join(dir, language)
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
    orthography = os.path.join(script_outdir, "orthography.py")
    output = os.path.join(script_outdir, "handbook.txt")
    output_full = os.path.join(script_outdir, "handbook_full.txt")
    lexicon = os.path.join(script_outdir, "lexicon_orth.tsv")
    sample_text = os.path.join(script_outdir, f"{STORY.value}.jsonl")
    user_prompt_dump = f"{script_outdir}/user_prompt_handbook.txt"

    # v New version with Morphosyntax params
    cmd = " \\\n".join(
        [
            CREATE,
            common_args,
            f'   --output="{output}"',
            f'   --output_full="{output_full}"',
            f'   --lexicon="{lexicon}"',
            f'   --sample_text="{sample_text}"',
            f'   --input_phonotactics="{phonotactics}"',
            f'   --orthography_code="{orthography}"',
            f'   --morphology_type="{morphology_type}"',
            f'   --main_word_order="{main_word_order}"', # ok
            f'   --adj_noun_word_order="{adj_noun_word_order}"', # ok
            f'   --posspron_noun_word_order="{posspron_noun_word_order}"', # ok
            f'   --num_noun_word_order="{num_noun_word_order}"', # ok
            f'   --adposition_noun_word_order="{adposition_noun_word_order}"', # ok
            # f'   --alignment="{alignment}"',
            f'   --case="{case}"', # ok
            f'   --definiteness="{definiteness}"', # ok
            f'   --adjective_agreement="{adjective_agreement}"', # ok
            f'   --nominal_number="{nominal_number}"', # ok
            f'   --comparative="{comparative}"', # ok
            f'   --tense_aspect_marking="{tense_aspect}"', # ok
            f'   --person="{person}"', # ok
            f'   --mood="{mood}"', # ok
            f'   --voice="{voice}"', # ok
            f'   --relativization="{relativization}"', # ok
            f'   --infinitive="{infinitive}"', # ok
            f'   --negation="{negation}"', # ok
            f'   --gender_marking="NONE"', # temporarily not used (as of Sep 18, 2025)
            f'   --inclusive_exclusive={inclusive_exclusive}',
            f'   --pro_drop={pro_drop}', # currently not used (as of Sep 18, 2025)
            f'   --user_prompt_dump="{user_prompt_dump}"',
        ]
    )

    # v Old version
    """"
    cmd = " \\\n".join(
        [
            CREATE,
            common_args,
            f'   --output="{output}"',
            f'   --output_full="{output_full}"',
            f'   --lexicon="{lexicon}"',
            f'   --sample_text="{sample_text}"',
            f'   --input_phonotactics="{phonotactics}"',
            f'   --orthography_code="{orthography}"',
            f'   --number_marking="{number_marking}"',
            f'   --case_marking="{case_marking}"',
            f'   --tense_aspect_marking="{tense_aspect_marking}"',
            f'   --head_marking="{head_marking}"',
            f'   --person_agreement="{person_agreement}"',
            f'   --main_word_order="{main_word_order}"',
            f'   --adj_noun_word_order="{adj_noun_word_order}"',
            f'   --adposition_noun_word_order="{adposition_noun_word_order}"',
            f'   --gender_marking="{gender_marking}"',
            f'   --numeral_classifier="{numeral_classifier}"',
            f'   --inclusive_exclusive="{inclusive_exclusive}"',
            f'   --user_prompt_dump="{user_prompt_dump}"',
        ]
    )
    """
    print(cmd)

    if RUN_COMMANDS.value:
        failure = os.system(cmd)
        if failure:
            with open(
                os.path.join(
                    script_outdir,
                    "handbook_fail.txt",
                ),
                "w",
            ) as stream:
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
                run_handbook_experiment(language=language,
                                        morphosyntax_language=MORPHOSYNTAX_LANGUAGE.value,
                                        script=script,
                                        dir=dir)
                if ONE_GRAMMAR_OUTPUT_ONLY.value:
                    break


if __name__ == "__main__":
    app.run(main)
