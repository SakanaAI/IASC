"""Prompt the LLM to design some aspect of the language."""
import sys
import os

sys.path.append(os.path.abspath("."))

import create_lib as lib

from absl import app
from absl import flags
from absl import logging
from llm import llm

# from llm.utils import load_system_instructions
from utils import common_utils as cu
from typing import Dict

LEGAL_TASKS = [
    # Prequel
    "story_composition",
    "sentence_design",
    # Initial set
    # Note that the phonotactics and phonotactics_improvement are "legacy"
    # versions. The newer code is under agentic_phonology/
    "phonotactics",  # THESE TWO TOGETHER
    "phonotactics_improvement",  # IN SEQUENCE
    "orthography",
    "modular_morphosyntax",
    "modular_morphosyntax_improvement",
    "cumulative_morphosyntax",
    "morphosyntax",
    "handbook",
    "new_translation",
]

MODULAR_MORPHOSYNTAX_TASKS = [
    "modular_morphosyntax",
    "modular_morphosyntax_improvement",
]

TASK = flags.DEFINE_list(
    "task",
    [],
    "Which tasks to perform: final one will loop until num_iterations.",
)
USER_PROMPT_DUMP = flags.DEFINE_string(
    "user_prompt_dump",
    "",
    "Optional list of path (base names) to dump user prompts to",
)

OUTPUT = cu.OUTPUT
OUTPUT_FULL = cu.OUTPUT_FULL
USER_PROMPT = cu.USER_PROMPT
NUM_ITERATIONS = cu.NUM_ITERATIONS


def main(unused_argv):
    """The main function to run the language creation tasks."""
    multitask = len(TASK.value) > 1
    assert len(TASK.value) == len(USER_PROMPT.value)
    for task in TASK.value:
        assert task in LEGAL_TASKS
    if not len(TASK.value):
        sys.exit()
    elif len(TASK.value) == 1: # just one task, which will be stored in `final_task`
        first_tasks = []
        first_user_prompts = []
    else:
        first_tasks = TASK.value[:-1]
        first_user_prompts = USER_PROMPT.value[:-1]
    final_task = TASK.value[-1]
    final_user_prompt = USER_PROMPT.value[-1]
    client = llm.client()

    def choose_params(task: str,
                      last_attempt: str = "",
                      which_iteration: int = 0):
        """Choose the parameters for the task."""
        if task == "phonotactics":
            params = cu.phonotactics_params()
        elif task == "phonotactics_improvement":
            params = cu.phonotactics_improvement_params(
                input_phonotactics=last_attempt,
                which_iteration=which_iteration,
            )
        elif task == "phonological_rules":
            params = cu.phonological_rules_params()
        elif task == "wals_morphosyntax":
            params = cu.wals_morphosyntax_params()
        elif task in MODULAR_MORPHOSYNTAX_TASKS:
            params = cu.modular_morphosyntax_params(
                previous_translation=last_attempt,
                which_iteration=which_iteration,
            )
        elif task == "cumulative_morphosyntax":
            params = cu.cumulative_morphosyntax_params()
        elif task == "morphosyntax":
            params = cu.morphosyntax_params()
        elif task == "story_composition":
            params = cu.story_params()
        elif task == "sentence_design":
            params = cu.sentence_params()
        elif task == "orthography":
            params = cu.orthography_params()
        elif task == "handbook":
            params = cu.handbook_params(filter_lexicon=False)
        elif task == "new_translation":
            params = cu.new_translation_params()
        else:
            params = {}
        return params

    def add_iter(path, itr):
        """Add the iteration number to the path."""
        if not multitask or not path:
            return path
        split = path.split(".")
        prefix = ".".join(split[:-1]) + f"_{itr}"
        suffix = split[-1]
        return prefix + "." + suffix

    itr = 0
    last_attempt = ""
    for j, task in enumerate(first_tasks): # if not multitask, this will be skipped
        params = choose_params(task, last_attempt)
        print("params:", params)
        user_prompt_path = first_user_prompts[j]
        output_path = add_iter(OUTPUT.value, itr)
        output_path_full = add_iter(OUTPUT_FULL.value, itr)
        os.makedirs(
            os.path.dirname(output_path), exist_ok=True
        )  # Chihiro: to avoid FileNotFoundError
        os.makedirs(os.path.dirname(output_path_full), exist_ok=True)
        user_prompt_dump = add_iter(USER_PROMPT_DUMP.value, itr)
        _ = lib.run_model(
            client=client,
            params=params,
            output_path=output_path,
            output_path_full=output_path_full,
            user_prompt_path=user_prompt_path,
            modular_morphosyntax=task in MODULAR_MORPHOSYNTAX_TASKS,
            user_prompt_dump=user_prompt_dump,
        )
        itr += 1
        last_attempt = output_path  ## SEE TODO ABOVE

    task = final_task
    user_prompt_path = final_user_prompt
    proceed = True
    while proceed and itr < NUM_ITERATIONS.value:
        if itr:
            logging.info(
                f"Next iteration will use last_attempt={last_attempt}",
            )
        logging.info(f"itr={itr}")

        params = choose_params(task, last_attempt, which_iteration=itr)
        print("Params:", params)
        output_path = add_iter(OUTPUT.value, itr)
        output_path_full = add_iter(OUTPUT_FULL.value, itr)
        user_prompt_dump = add_iter(USER_PROMPT_DUMP.value, itr)
        logging.info(f"output_path={output_path}")

        proceed = lib.run_model(
            client=client,
            params=params,
            output_path=output_path,
            output_path_full=output_path_full,
            user_prompt_path=user_prompt_path,
            modular_morphosyntax=task in MODULAR_MORPHOSYNTAX_TASKS,
            user_prompt_dump=user_prompt_dump,
        )
        itr += 1
        last_attempt = output_path
    logging.info("Done")


if __name__ == "__main__":
    app.run(main)
