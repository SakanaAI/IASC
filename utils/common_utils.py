"""Common flags and utilities for prompt construction."""

import csv
import jsonlines
import os
import random
import re
import subprocess

from absl import flags
from absl import logging
from collections import defaultdict
from corpus.corpus_management import text_to_interlinear
from importlib.machinery import SourceFileLoader
from jinja2 import Template
from types import ModuleType
from typing import Dict, Any, List
from pydantic import BaseModel
import re
import ast

OUTPUT = flags.DEFINE_string("output", None, "Output code or data.")
OUTPUT_FULL = flags.DEFINE_string("output_full", None, "Full output from LLM.")
NUM_ITERATIONS = flags.DEFINE_integer(
    "num_iterations",
    4,
    "Number of iterations of improvement.",
)
SYSTEM_PROMPT = flags.DEFINE_string(
    "system_prompt",
    "prompts/system.txt",
    "System prompt.",
)
USER_PROMPT = flags.DEFINE_list(
    "user_prompt",
    [],
    "User prompts.",
)
INPUT_PHONOTACTICS = flags.DEFINE_string(
    "input_phonotactics",
    None,
    "Input phonotactics file.",
)
LANGUAGE = flags.DEFINE_string(
    "language",
    None,
    "Language to mimic",
)
STRESS_PLACEMENT = flags.DEFINE_enum(
    "stress_placement",
    None,
    ["final", "penultimate", "antepenultimate", "initial"],
    "Syllable for stress placement",
)
## These go with the --task=morphosyntax setting.
##
## TODO(rws): This might be better handled with a json file or collection of
## json files.
NUMBER_MARKING = flags.DEFINE_list(
    "number_marking",
    ["-SING", "-PLUR"],
    "Number marking pattern for the language.",
)
CASE = flags.DEFINE_string(
    "case",
    "case=['nominative', 'accusative'] case_marking='suffix'",
    "Case strategies used in the language.",
)
TENSE_ASPECT_MARKING = flags.DEFINE_list(
    "tense_aspect_marking",
    ["PRES-", "PAST-", "FUT-", "-PERF"],
    "Tense/aspect marking pattern for the language.",
)
HEAD_MARKING = flags.DEFINE_list(
    "head_marking",
    ["VERG-", "VABS-"],
    "Verbal head (agreement) marking for the language.",
)
PERSON_AGREEMENT = flags.DEFINE_list(
    "person_agreement",
    [
        "1SGERG-",
        "2SGERG-",
        "3SGERG-",
        "1PLERG-",
        "2PLERG-",
        "3PLERG-",
        "-1SGABS",
        "-2SGABS",
        "-3SGABS",
        "-1PLABS",
        "-2PLABS",
        "-3PLABS",
    ],
    "Verbal head person-agreement marking for the language.",
    # DEPRECATED!
)
PERSON = flags.DEFINE_string(
    "person",
    "person_agreement=['first', 'second', 'third'] person_marking_strategy='suffix' verbal_number_agreement=['singular', 'plural'] verbal_number_marking_strategy='suffix'",
    "Verbal head person-agreement marking for the language.",
)
MAIN_WORD_ORDER = flags.DEFINE_string(
    "main_word_order",
    "VSO",
    "Main word ordering.",
)
ADJ_NOUN_WORD_ORDER = flags.DEFINE_string(
    "adj_noun_word_order",
    "NA",
    "Adjective-noun word ordering.",
)
POSSPRON_NOUN_WORD_ORDER = flags.DEFINE_string(
    "posspron_noun_word_order",
    "PossN",
    "Possessive pronoun-noun word ordering.",
)
NUM_NOUN_WORD_ORDER = flags.DEFINE_string(
    "num_noun_word_order",
    "NumN",
    "Numeral-noun word ordering.",
)
ADPOSITION_NOUN_WORD_ORDER = flags.DEFINE_string(
    "adposition_noun_word_order",
    "PN",
    "Adposition-noun word ordering.",
)
MORPHOLOGY_TYPE = flags.DEFINE_string(
    "morphology_type",
    "agglutinative",
    "Morphology type of the language.",
)
GENDER_MARKING = flags.DEFINE_list(
    "gender_marking",
    ["-MASC", "-FEM", "-NEUT"],
    "Nominal gender marking.",
)
NUMERAL_CLASSIFIER = flags.DEFINE_string(
    "numeral_classifier",
    "HAS_CLASSIFIERS",
    "Whether the language has numeral classifiers or not.",
)
INCLUSIVE_EXCLUSIVE = flags.DEFINE_string(
    "inclusive_exclusive",
    "INCLUSIVE_EXCLUSIVE",
    "Whether the language has the inclusive-exclusive distinction or not.",
)
DEFINITENESS = flags.DEFINE_string(
    "definiteness",
    "definiteness=['definite', 'indefinite'] definiteness_marking_strategy='prepositional word', definiteness_agreement='number'",
    "Whether the language has the definite-indefinite distinction or not.",
)
DEFINITENESS_MARKING_STRATEGY = flags.DEFINE_string(
    "definiteness_marking_strategy",
    "prepositional word",
    "The strategy used for definiteness marking, such as prefix, suffix, prepositional word, or postpositional word.",
)
PRO_DROP = flags.DEFINE_string(
    "pro_drop",
    "pro_drop",
    "Whether the language is a pro-drop language or not.",
)
ADJECTIVE_AGREEMENT = flags.DEFINE_string(
    "adjective_agreement",
    "adjective_agreement=['gender', 'number'] adjective_agreement_strategy='suffix'",
    "List of features that adjectives agree with in the language, such as gender, number, and case. If None, no adjective agreement is used.",
)
NEGATION = flags.DEFINE_string(
    "negation",
    "prepositional word",
    "The strategy used for negation marking, such as prefix, suffix, prepositional word, or postpositional word.",
)
VOICE = flags.DEFINE_string(
    "voice",
    "voice=['active', 'passive'] voice_marking='suffix'",
    "List of voices used in the language, such as active and passive. If None, no voice marking is used.",
)
MOOD = flags.DEFINE_string(
    "mood",
    "mood=['indicative', 'imperative'] mood_marking='suffix'",
    "List of moods used in the language, such as indicative, subjunctive, and imperative. If None, no mood marking is used.",
)
RELATIVIZATION = flags.DEFINE_string(
    "relativization",
    "relativization_order='head-initial' relativization_marking='head-marking' relativizer_position='prepositional' relativizer_morpheme='word'",
    "Relativization strategies used in the language.",
)
INFINITIVE = flags.DEFINE_string(
    "infinitive",
    "infinitive='infinitive' infinitive_marking_strategy='suffix'",
    "Infinitive strategies used in the language.",
)
NOMINAL_NUMBER = flags.DEFINE_string(
    "nominal_number",
    "nominal_number=['singular', 'plural'] nominal_number_marking_strategy='suffix'",
    "Nominal number marking used in the language.",
)
COMPARATIVE = flags.DEFINE_string(
    "comparative",
    "comparative=['comparative', 'superlative'] comparative_marking_strategy='suffix'",
    "Comparative and superlative marking used in the language.",
)

MORPHOSYNTAX_MODULES = flags.DEFINE_list(
    "morphosyntax_modules",
    ["word_order"],
    "Modules to use for modular morphosyntax instruction",
)
SAMPLE_TEXT = flags.DEFINE_string(
    "sample_text",
    None,
    "Path to sample text to `translate`.",
)
PREVIOUS_TRANSLATION = flags.DEFINE_string(
    "previous_translation",
    None,
    "Path to the previous `translation` in the cumulative translation approach",
)

# The following goes with the instructions for story composition in story.txt.
STORY_TOPIC = flags.DEFINE_string(
    "story_topic",
    None,
    "Topic for a short story composed by the LLM.",
)
# The following goes with the instructions for sentence_design
NUMBER_OF_SENTENCES = flags.DEFINE_integer(
    "number_of_sentences",
    10,
    "Number of sentences to compose for testing grammatical properties.",
)
# The following goes with the instructions for orthography construction in
# orthography.txt, and also with the handbook.
LEXICON = flags.DEFINE_string(
    "lexicon",
    None,
    "A lexicon with words and their pronunciations.",
)
SCRIPT = flags.DEFINE_string(
    "script",
    "Latin",
    "An alphabetic script such as Latin or Cyrillic",
)
# The following goes with the handbook
ORTHOGRAPHY_CODE = flags.DEFINE_string(
    "orthography_code",
    None,
    "Path to the python code for the orthography",
)
# For the new translation task
NEW_SAMPLE_TEXT = flags.DEFINE_string(
    "new_sample_text",
    None,
    "Path to new sample text to `translate`.",
)
TRANSLATION = flags.DEFINE_string(
    "translation",
    None,
    "Path to prior `translation` of prior sample.",
)
HANDBOOK = flags.DEFINE_string(
    "handbook",
    None,
    "Path to handbook.",
)

# regex for detecting comments in the system prompt texts
COMMENT = re.compile("<!--.*?-->", re.DOTALL)


def load_system_instructions(system_instructions: str) -> str:
    """Loads system instructions from file.

    Args:
        system_instructions: A path.
        verbose: boolean, whether to print out the instructions.
    Returns:
        A string containing the system instructions.
    """
    with open(system_instructions) as stream:
        instructions = stream.readlines()
        instructions = "".join(instructions)
        instructions = instructions.replace("<INSTRUCTIONS>", "")
        instructions = instructions.replace("</INSTRUCTIONS>", "")
        instructions = COMMENT.sub("", instructions)
        instructions = instructions.strip()
        return instructions


def create_system_prompt() -> str:
    return load_system_instructions(SYSTEM_PROMPT.value)


feature_to_file = {
    "inclusive_exclusive.txt": "inclusive_exclusive.txt",
    "case_marking.txt": "nominal_case_marking.txt",
    "number.txt": "nominal_number_marking.txt",
    "person_agreement.txt": "verbal_person_agreement.txt",
    "tense_aspect.txt": "verbal_tense_aspect_marking.txt",
    "word_order.txt": "word_order.txt",
    "negation.txt": "negation.txt",
    "relativization.txt": "relativization.txt",
    "mood.txt": "mood.txt",
    "voice.txt": "voice.txt",
    "infinitive.txt": "infinitive.txt",
    "extras.txt": "extras.txt",
}


def create_user_prompt(
    params: Dict[str, str],
    user_prompt_path: str = "",
    modular_morphosyntax: bool = False,
) -> str:
    """Constructs the user prompt.

    Args:
        params: A dictionary, parameters for the user prompt.
        user_prompt_path: Path to user prompt.
        modular_morphosyntax: Whether or not the prompt constructor should
        use the modular_morphosyntax constructor.
    Returns:
        Fully constructed prompt string.
    """
    if not user_prompt_path:
        user_prompt_basename = os.path.basename(USER_PROMPT.value[0])
        user_prompt_dirname = os.path.dirname(USER_PROMPT.value[0])
        print(user_prompt_basename)
        user_prompt_path = feature_to_file.get(
            user_prompt_basename, user_prompt_basename
        )
        user_prompt_path = os.path.join(user_prompt_dirname, user_prompt_path)
        user_prompt = load_system_instructions(user_prompt_path)
    elif modular_morphosyntax:
        user_prompt_basename = os.path.basename(
            user_prompt_path
        )  # e.g. basename("folder/file.txt") -> "file.txt"
        user_prompt_dirname = os.path.dirname(user_prompt_path)
        user_prompt_path = feature_to_file.get(
            user_prompt_basename, user_prompt_basename
        )
        user_prompt_path = os.path.join(user_prompt_dirname, user_prompt_path)
        user_prompt = modular_morphosyntax_prompt(user_prompt_path)
    else:
        user_prompt_basename = os.path.basename(user_prompt_path)
        user_prompt_dirname = os.path.dirname(user_prompt_path)
        user_prompt_path = feature_to_file.get(
            user_prompt_basename, user_prompt_basename
        )
        user_prompt_path = os.path.join(user_prompt_dirname, user_prompt_path)
        user_prompt = load_system_instructions(user_prompt_path)
    prompt = Template(user_prompt).render(params).strip()
    print(prompt)

    prompt_token_count = int(len(prompt) / 4)  # Rough estimate
    logging.info(f"prompt_token_count≈{prompt_token_count}")
    if prompt_token_count > 4096:
        logging.warning(f"prompt_token_count≈{prompt_token_count} may be too large")
    return prompt


def load_python_source_module(path: str) -> ModuleType:
    """Loads Python source from path as a module.

    **NB**: This is potentially dangerous. Also load_module() is deprecated so we
    need to investigate how to use exec_module() and create_module(), which of
    course have a different interface.

    Args:
        path: path to the predefined Python code.
    Returns:
        A module.
    """
    return SourceFileLoader("python_source", path).load_module()


def phonological_rules_params():
    assert INPUT_PHONOTACTICS.value is not None
    assert STRESS_PLACEMENT.value is not None
    __phonotactics__ = load_python_source_module(INPUT_PHONOTACTICS.value)
    consonants = ", ".join([c for c in __phonotactics__.consonants.keys()])
    vowels = ", ".join([c for c in __phonotactics__.vowels.keys()])
    return {
        "consonants": consonants,
        "vowels": vowels,
        "stress_placement": STRESS_PLACEMENT.value,
    }


def phonotactics_params():
    assert LANGUAGE.value is not None
    return {
        "language": LANGUAGE.value,
    }


def phonotactics_improvement_params(
    input_phonotactics: str = "", which_iteration: int = 0
) -> Dict[str, Any]:
    assert LANGUAGE.value is not None
    if not input_phonotactics:
        assert INPUT_PHONOTACTICS.value is not None
        input_phonotactics = INPUT_PHONOTACTICS.value
    with open(input_phonotactics) as stream:
        previous_code = stream.read().strip()
    cmd = ["python3", input_phonotactics, "--num_morphemes=100"]
    process_output = subprocess.run(
        cmd,
        cwd=".",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if "Error" in process_output.stderr:
        output_sample = "\n".join(
            [
                "Oops! It looks as if you have a bug in your previous code:",
                process_output.stderr,
                "Before proceeding, you will need to fix this bug.",
            ]
        )
    else:
        output_sample = process_output.stdout
    return {
        "language": LANGUAGE.value,
        "previous_code": previous_code,
        "output_sample": output_sample,
        "num_iterations": NUM_ITERATIONS.value,
        "which_iteration": which_iteration,
    }


def morphosyntax_placeholder_table() -> Dict[str, str]:
    """Returns a dictionary with morphosyntax placeholder values.

    This function constructs a dictionary with morphosyntactic parameters
    using the values defined in the flags. It ensures that if PERSON_AGREEMENT
    is specified, it takes precedence over HEAD_MARKING for the head_marking
    value.
    """
    # PERSON_AGREEMENT takes precedence over HEAD_MARKING
    if PERSON_AGREEMENT.value:
        head_marking = " ".join(PERSON_AGREEMENT.value)
        logging.info(
            "Using PERSON_AGREEMENT for head_marking since that is specified.",
        )
    else:
        head_marking = " ".join(HEAD_MARKING.value) or "NONE"
        logging.info("Using HEAD_MARKING for head_marking")
    return {
        "number_marking": " ".join(NUMBER_MARKING.value) or "NONE",
        "case_marking": " ".join(CASE.value) or "NONE",
        "tense_aspect": " ".join(TENSE_ASPECT_MARKING.value) or "NONE",
        "head_marking": head_marking,
        "main_word_order": MAIN_WORD_ORDER.value or "NONE",
        "adj_noun_word_order": ADJ_NOUN_WORD_ORDER.value or "NONE",
        "posspron_noun_word_order": POSSPRON_NOUN_WORD_ORDER.value or "NONE",
        "num_noun_word_order": NUM_NOUN_WORD_ORDER.value or "NONE",
        "adposition_noun_word_order": ADPOSITION_NOUN_WORD_ORDER.value or "NONE",
        "gender_marking": " ".join(GENDER_MARKING.value) or "NONE",
        "negation": NEGATION.value or "NONE",
        "inclusive_exclusive": INCLUSIVE_EXCLUSIVE.value or "NONE",
        "numeral_classifier": NUMERAL_CLASSIFIER.value or "NONE",
        "definiteness": " ".join(DEFINITENESS.value) or "NONE",
    }


def morphosyntax_params() -> Dict[str, str]:
    """Returns a dictionary with morphosyntactic parameters."""
    assert SAMPLE_TEXT.value is not None
    params = morphosyntax_placeholder_table()
    text = open(SAMPLE_TEXT.value).read().strip()
    params["text"] = text
    return params


def story_params() -> Dict[str, str]:
    return {
        "story_topic": STORY_TOPIC.value,
    }


def sentence_params() -> Dict[str, str]:
    general_guidelines = load_system_instructions(
        "prompts/sentence_design/general_guidelines.txt",
    )
    return {
        "number_of_sentences": NUMBER_OF_SENTENCES.value,
        "general_guidelines": general_guidelines,
    }


# TODO(rws): Use the Lexicon class code instead
def _load_lexicon(path: str) -> Dict[str, str]:
    lexicon = defaultdict(str)
    with open(path) as s:
        reader = csv.reader(s, delimiter="\t", quotechar='"')
        for row in reader:
            lexicon[row[0]] = row[1]
    return lexicon


def orthography_params():
    assert LEXICON.value is not None
    assert INPUT_PHONOTACTICS.value is not None
    __phonotactics__ = load_python_source_module(INPUT_PHONOTACTICS.value)
    consonants = ", ".join([c for c in __phonotactics__.consonants.keys()])
    vowels = ", ".join([c for c in __phonotactics__.vowels.keys()])
    lexicon = _load_lexicon(LEXICON.value)
    forms = []
    # Collect a sample of longer forms.
    for form in lexicon.values():
        if len(form.split()) > 2:
            forms.append(form)
    random.shuffle(forms)
    lexicon_sample = "\n".join(forms[:30])
    return {
        "consonants": consonants,
        "vowels": vowels,
        "lexicon_sample": lexicon_sample,
        "script": SCRIPT.value,
    }


def handbook_params(filter_lexicon=True):
    assert LEXICON.value is not None
    assert INPUT_PHONOTACTICS.value is not None
    assert ORTHOGRAPHY_CODE.value is not None
    assert SAMPLE_TEXT.value is not None
    __phonotactics__ = load_python_source_module(INPUT_PHONOTACTICS.value)
    consonants = ", ".join([c for c in __phonotactics__.consonants.keys()])
    vowels = ", ".join([c for c in __phonotactics__.vowels.keys()])

    if SAMPLE_TEXT.value.endswith(".jsonl"):
        with jsonlines.open(SAMPLE_TEXT.value) as reader:
            sample_text = text_to_interlinear(reader, include_source=False)
    else:
        sample_text = open(SAMPLE_TEXT.value).read().strip()

    # Kludgy + deprecated
    def extract_needed_headwords(text):
        text = text.split("\n")
        hdr = -1
        needed = set()
        for line in text:
            line = line.strip()
            if line.startswith("====="):
                hdr = 2
            elif hdr == 0:
                glosses = line.split()
                for gloss in glosses:
                    morphs = gloss.split("-")
                    for morph in morphs:
                        needed.add(morph)
            hdr -= 1
        return needed

    lexicon_table = _load_lexicon(LEXICON.value)
    lexicon = []
    if filter_lexicon:
        needed_headwords = extract_needed_headwords(sample_text)
        # Include all grammatical features (o is uppercase) and other needed headwords
        # given the story:
        for i, o in lexicon_table.items():
            if o.isupper():
                lexicon.append((i, o))
            elif o in needed_headwords:
                lexicon.append((i, o))
    else:
        for i, o in lexicon_table.items():
            lexicon.append((i, o))
    lexicon.sort()
    lexicon = "\n".join([f"{i}\t{o}" for (i, o) in lexicon])

    if not SAMPLE_TEXT.value.endswith(".jsonl"):
        # Remove the English translation from the sample text!
        text = sample_text.split("\n")

        hdr = -1
        sample_text = []
        for line in text:
            line = line.strip()
            if line.startswith("====="):
                sample_text.append(line)
                hdr = 1
            elif hdr == 0:
                pass
            else:
                sample_text.append(line)
            hdr -= 1
        sample_text = "\n".join(sample_text)

    with open(ORTHOGRAPHY_CODE.value) as s:
        orthography_code = s.read().strip()
    table = morphosyntax_placeholder_table()
    table["consonants"] = consonants
    table["vowels"] = vowels
    table["lexicon"] = lexicon
    table["orthography_code"] = orthography_code
    table["sample_text"] = sample_text
    return table


def new_translation_params(filter_lexicon=True):
    assert SAMPLE_TEXT.value is not None
    assert NEW_SAMPLE_TEXT.value is not None
    assert TRANSLATION.value is not None
    assert HANDBOOK.value is not None
    with open(SAMPLE_TEXT.value) as s:
        sample_text = s.read().strip()
    with open(TRANSLATION.value) as s:
        translation = s.read().strip()
    with open(NEW_SAMPLE_TEXT.value) as s:
        new_sample_text = s.read().strip()
    with open(HANDBOOK.value) as s:
        handbook = s.read().strip()
    for line in handbook.split("\n"):
        if line.startswith("TITLE:"):
            language_name = line.split()[-1]
            break
    return {
        "language_name": language_name,
        "sample_text": sample_text,
        "translation": translation,
        "new_sample_text": new_sample_text,
        "handbook": handbook,
    }


###############################################################################
# Modular morphosyntax prompt construction
###############################################################################

# TODO(ct): This dict is not used anymore?
MORPHOSYNTAX_MODULES_TABLE = defaultdict(
    str,
    {
        "inclusive_exclusive": "inclusive_exclusive.txt",
        "negation": "negation.txt",
        "nominal_case_marking": "nominal_case_marking.txt",
        "case": "nominal_case_marking.txt",
        "nominal_gender": "nominal_gender.txt",
        "nominal_number_marking": "nominal_number_marking.txt",
        "numeral_classifier": "numeral_classifier.txt",
        "definiteness": "definiteness.txt",
        "definiteness_marking_strategy": "definiteness_marking_strategy.txt",
        "verbal_head_marking": "verbal_head_marking.txt",
        "verbal_person_agreement": "verbal_person_agreement.txt",
        "verbal_tense_aspect_marking": "verbal_tense_aspect_marking.txt",
        "word_order": "word_order.txt",
    },
)


# TODO(rws): I think there was a bug here since the previous_translation is a
# path name to the previous translation, but that text needs to be loaded.
def modular_morphosyntax_params(previous_translation="", which_iteration=0):
    assert SAMPLE_TEXT.value is not None
    if PERSON_AGREEMENT.value:
        head_marking = " ".join(PERSON_AGREEMENT.value)
        logging.info(
            "Using PERSON_AGREEMENT for head_marking since that is specified.",
        )
    else:
        head_marking = " ".join(HEAD_MARKING.value) or "NONE"
        logging.info("Using HEAD_MARKING for head_marking")
    text = open(SAMPLE_TEXT.value).read().strip()
    return {
        "adj_noun_word_order": ADJ_NOUN_WORD_ORDER.value or "NONE",
        "adposition_noun_word_order": ADPOSITION_NOUN_WORD_ORDER.value or "NONE",
        "case_marking": " ".join(CASE.value) or "NONE",
        "gender_marking": " ".join(GENDER_MARKING.value) or "NONE",
        "definiteness": " ".join(DEFINITENESS.value) or "NONE",
        "definiteness_marking_strategy": DEFINITENESS_MARKING_STRATEGY.value or "NONE",
        "inclusive_exclusive": INCLUSIVE_EXCLUSIVE.value or "NONE",
        "head_marking": head_marking,
        "main_word_order": MAIN_WORD_ORDER.value or "NONE",
        "number_marking": " ".join(NUMBER_MARKING.value) or "NONE",
        "numeral_classifier": NUMERAL_CLASSIFIER.value or "NONE",
        "tense_aspect": " ".join(TENSE_ASPECT_MARKING.value) or "NONE",
        "text": text,
        "negation": NEGATION.value or "NONE",
        "previous_translation": previous_translation,
        "which_iteration": which_iteration,
        "num_iterations": NUM_ITERATIONS.value,
    }


def modular_morphosyntax_prompt(morphosyntax_main_path: str) -> str:
    """
    Constructs the modular morphosyntax prompt.

    Args:
        morphosyntax_main_path: Path to the main morphosyntax prompt file.
    Returns:
        A string containing the modular morphosyntax prompt.
    """
    msg = "You must at least specify word order."
    assert "word_order" in MORPHOSYNTAX_MODULES.value, msg
    # Put word_order last so as not to mix it up with morphosyntactic
    # marking. Also make sure they are unique:
    modules = set(MORPHOSYNTAX_MODULES.value)
    double_head_marking = (
        "verbal_head_marking" in modules and "verbal_person_agreement" in modules
    )
    msg = "You cannot have both `verbal_head_marking` and `verbal_person_agreement`"
    assert not double_head_marking, msg
    modules = [m for m in modules if m != "word_order"]
    modules.append("word_order")
    modules = [
        MORPHOSYNTAX_MODULES_TABLE[k] for k in modules if MORPHOSYNTAX_MODULES_TABLE[k]
    ]
    prompt = load_system_instructions(morphosyntax_main_path)
    morphosyntax_text = []
    delim = "==============\n"
    for line in prompt.split("\n"):
        if line.strip() == "**MODULES**":
            base = os.path.dirname(morphosyntax_main_path)
            module_texts = [delim]
            for i, module in enumerate(modules):
                path = os.path.join(base, "morphosyntax", module)
                module_text = load_system_instructions(path)
                section_number = f"{i + 1}"
                # We can't use format here since this will mess up the formatting
                # of {{}} for the Template later on.
                module_text = module_text.replace(
                    "{section_number}",
                    section_number,
                )
                module_texts.append(module_text)
                module_texts.append(delim)
            morphosyntax_text.append("\n".join(module_texts))
        else:
            morphosyntax_text.append(line)
    morphosyntax_text = "\n".join(morphosyntax_text)
    return morphosyntax_text


# For the cumulative morphosyntax approach:
def cumulative_morphosyntax_params():
    assert SAMPLE_TEXT.value is not None
    if PERSON_AGREEMENT.value:
        head_marking = " ".join(PERSON_AGREEMENT.value)
        logging.info(
            "Using PERSON_AGREEMENT for head_marking since that is specified.",
        )
    else:
        head_marking = " ".join(HEAD_MARKING.value) or "NONE"
        logging.info("Using HEAD_MARKING for head_marking")
    original_text = open(SAMPLE_TEXT.value).read().strip()
    if PREVIOUS_TRANSLATION.value:
        with open(PREVIOUS_TRANSLATION.value) as stream:
            previous_translation = stream.read().strip()
    else:
        previous_translation = ""

    def load_dict_value(value: str | Any) -> Dict[str, Any]:
        """Loads a dict value from a string representation."""
        if not value:
            return None
        elif isinstance(value, list):
            return value

        try:
            if isinstance(eval(value), dict):
                # if the value is already a dictionary, return it directly
                return eval(value)
        except Exception as e:
            # if the value cannot be parsed by eval, we need to reformat it
            # to a dictionary format
            return parse_str_to_dict(value)

    def format_params(absl_flag, param_key: str, default_value: str = "NONE"):
        """Formats the parameters for a given absl flag."""
        if absl_flag.value != "None" or absl_flag.value:
            return load_dict_value(absl_flag.value)
        else:
            return {param_key: default_value}

    case_params = format_params(CASE, "case", "NONE")
    adjective_agreement_params = format_params(ADJECTIVE_AGREEMENT, "adjective_agreement", "NONE")
    comparative_params = format_params(COMPARATIVE, "comparative", "NONE")
    relativization_params = format_params(RELATIVIZATION, "relativization", "NONE")
    voice_params = format_params(VOICE, "voice", "NONE")
    mood_params = format_params(MOOD, "mood", "NONE")
    infinitive_params = format_params(INFINITIVE, "infinitive", "NONE")
    definiteness_params = format_params(DEFINITENESS, "definiteness", "NONE")
    person_params = format_params(PERSON, "person", "NONE")
    number_params = format_params(NOMINAL_NUMBER, "nominal_number", "NONE")

    print("person_params:", person_params) # debug

    params = {  # Do these values just use the defaults? We might need to change this
        "adj_noun_word_order": ADJ_NOUN_WORD_ORDER.value or "NONE",
        "posspron_noun_word_order": POSSPRON_NOUN_WORD_ORDER.value or "NONE",
        "num_noun_word_order": NUM_NOUN_WORD_ORDER.value or "NONE",
        "adposition_noun_word_order": ADPOSITION_NOUN_WORD_ORDER.value or "NONE",
        # "person_agreement": " ".join(PERSON_AGREEMENT.value) or "NONE",
        "person": PERSON.value or "NONE",
        "adjective_agreement": ADJECTIVE_AGREEMENT.value or "NONE",
        "comparative": COMPARATIVE.value or "NONE",
        "case": CASE.value or "NONE",
        "gender_marking": " ".join(GENDER_MARKING.value) or "NONE",
        "pro_drop": PRO_DROP.value or "NONE",
        "definiteness": DEFINITENESS.value or "NONE",
        "definiteness_marking_strategy": DEFINITENESS_MARKING_STRATEGY.value or "NONE",
        "inclusive_exclusive": INCLUSIVE_EXCLUSIVE.value or "NONE",
        # "head_marking": head_marking,
        "main_word_order": MAIN_WORD_ORDER.value or "NONE",
        # "number_marking": " ".join(NUMBER_MARKING.value) or "NONE",
        "nominal_number": NOMINAL_NUMBER.value or "NONE",
        "numeral_classifier": NUMERAL_CLASSIFIER.value or "NONE",
        "tense_aspect": " ".join(TENSE_ASPECT_MARKING.value) or "NONE",
        "voice": VOICE.value or "NONE",
        "mood": MOOD.value or "NONE",
        "negation": NEGATION.value or "NONE",
        "relativization": RELATIVIZATION.value or "NONE",  # it's a BaseModel but expressed in string
        "original_text": original_text,
        "previous_translation": previous_translation,
    }

    # Update the params with the retrieved dict values
    params.update(case_params)
    params.update(adjective_agreement_params)
    params.update(comparative_params)
    params.update(relativization_params)
    params.update(voice_params)
    params.update(mood_params)
    params.update(infinitive_params)
    params.update(definiteness_params)
    params.update(person_params)
    params.update(number_params)

    return params


def parse_str_to_dict(input_str: str) -> Dict[str, Any]:
    """
    Parses a string representation of a BaseModel and returns an instance of the model.

    Args:
        input_str: The string representation of the BaseModel.
            It looks like `"category='example' value=1"` for `BaseModel(category='example', value=1)`.

    Returns:
        An instance of the BaseModel.
    """
    pattern = re.compile(r"(\w+)=((?:\[[^\]]*\])|(?:'[^']*')|(?:\"[^\"]*\")|\S+)")
    result = {}
    for key, value in pattern.findall(input_str):
        try:
            result[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            result[key] = value
    return result


def find_last_stage(stages: List[str],
                    output_files: List[str]) -> str:
    """Find the last morphosyntactic param stage.

    Args:
        stages: List of all possible stages.
        output_files: List of output files in the scripts directory.
    Returns:
        The last stage found in the scripts directory (e.g. `infinitive`).
    """
    script_basenames = [os.path.basename(f) for f in output_files]
    # strip extensions and indices
    script_stages = set([name.split("_")[0] for name in script_basenames])

    last_stage = None
    for stage in stages:
        if stage in script_stages:
            last_stage = stage
    if last_stage is None:
        raise ValueError("No stages found in the scripts directory.")

    return last_stage
