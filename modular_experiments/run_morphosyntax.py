"""Runner for morphosyntactic series.

The first task is setting the word order, for which we set NUM_ITER iterations
for each configuration. For example, given with four configurations defined in
morphosyntax_params, and NUM_ITER == 3, that will give us k=12 languages, 3 for
each configuration setting.

Then for each subsequent morphosyntactic stage we iterate over these, and for
each of the settings iterate over the settings for the new parameters.

This code does not actually run the scripts it creates unless --run_commands is
set.

Edit by Chihiro:
- Use `argparse` instead of `absl` for command line arguments. (for my personal preference)
- Substitute PARAMS and SAFE_PARAMS with the params defined with `pydantic.BaseModel`.
- Change the indent style from 2 spaces to 4 spaces.
- Remove unused imports.
"""

import sys
import os
sys.path.append(os.path.abspath("."))
import copy
from jinja2 import Template
from typing import Any, Dict, Tuple, Optional
import argparse
import glob


# local imports
from modular_experiments import morphosyntax_params
from utils.common_utils import find_last_stage


def get_args() -> argparse.Namespace:
    """Get the command line arguments."""
    parser = argparse.ArgumentParser(description="Run morphosyntactic experiments.")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["claude", "gpt5nano", "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4o-mini", "gemini-2.5-flash", "gemini-2.5-pro"],
        default="claude",
        help="Model to use for the experiments.",
    )
    parser.add_argument(
        "--open_ai_api_key",
        type=str,
        default="",
        help="OpenAI API key for the model.",
    )
    parser.add_argument(
        "--modular_experiment_outputs",
        type=str,
        default="modular_experiment_outputs",
        help="Subdirectory for the experimental data",
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=3,
        help="Number of iterations of each configuration",
    )
    parser.add_argument(
        "-r",
        "--run_commands",
        action='store_true',
        help="Actually run the created commands.",
    )
    parser.add_argument(
        "--story",
        type=str,
        default="story_the_two_towers",  # Good for both dual and incl/excl.
        help="Name of the story",
    )
    parser.add_argument(
        "--storydir",
        type=str,
        default="llm_stories",
        help="Path to directory of stories.",
    )
    parser.add_argument(
        "--use_safe_params",
        action='store_true',
        help="Use the 'safe' parameters, which we think are likely to be most successful.",
    )
    parser.add_argument(
        "--metascript_dir",
        type=str,
        default="",
        help="Directory to store the metascripts.",
    )
    parser.add_argument(
        "--premade_params_language",
        type=str,
        default="turkish",
        help="Language to use for the premade parameters. Options: 'turkish', 'french', 'arabic', 'welsh', 'vietnamese', 'mizo', 'fijian', 'hixkaryana', 'ainu'.",
    )
    parser.add_argument(
        "--pipeline",
        action='store_true',
        help="Whether to run the evaluation pipeline after generating the outputs.",
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        required=False,
        help="Path to the reference glosses file for evaluation.",
    )
    parser.add_argument(
        "--do_review",
        action="store_true",
        help="If set, a review stage with few-shot in-context learning will be done at the end of the morphosyntax pipeline."
    )

    return parser.parse_args()


def story_path(story: str,
               storydir: str) -> str:
    """Creates the path to the story file.

    Args:
        story: Name of the story to use.
        storydir: Directory containing the stories.
    Returns:
        Path to the story file.
    """
    return os.path.join(storydir, f"{story}.txt")


COMMON_ARGS = Template("""    --model="{{model}}" \\
    --task="cumulative_morphosyntax" \\
    --sample_text="{{story}}" \\
    --open_ai_api_key={{open_ai_api_key}} \\
    --num_iterations=1"""
    )

CREATE = "python3 create/create.py"
LANGUAGE_TO_PARAMS = {
    "french": morphosyntax_params.sample_params_french,
    "turkish": morphosyntax_params.sample_params_turkish,
    "arabic": morphosyntax_params.sample_params_arabic,
    "welsh": morphosyntax_params.sample_params_welsh,
    "vietnamese": morphosyntax_params.sample_params_vietnamese,
    "mizo": morphosyntax_params.sample_params_mizo,
    "fijian": morphosyntax_params.sample_params_fijian,
    "hixkaryana": morphosyntax_params.sample_params_hixkaryana,
    "hard": morphosyntax_params.sample_params_hard,
    "ainu": morphosyntax_params.sample_params_ainu,
}
STAGES = [
    "inclusive_exclusive",
    "negation",
    "nominal_number",
    "definiteness",
    "adjective_agreement",
    "comparative",
    "case",
    "mood",
    "tense_aspect",
    "person",
    "voice",
    "relativization",
    "infinitive",
    "extras",
    "review"
]
STAGE_FLAGS = {
    "inclusive_exclusive": "inclusive_exclusive",
    "negation": "negation",
    "nominal_number": "nominal_number",
    "definiteness": "definiteness",
    "adjective_agreement": "adjective_agreement",
    "case": "case",
    "tense_aspect": "tense_aspect_marking",
    "mood": "mood",
    "person": "person",
    "voice": "voice",
    "relativization": "relativization",
    "infinitive": "infinitive",
    "comparative": "comparative",
    "extras": None,
    "review": "review",
}
# the values correspond to the file names.
# each stage needs to have a corresponding prompt txt file. Make sure to have it ready
assert set(STAGES) == set(STAGE_FLAGS.keys())


def create_common_args(open_ai_api_key: Optional[str],
                       model: str,
                       story: str,
                       storydir: str) -> str:
    """Creates the common set of arguments used by the experiments.

    Args:
        open_ai_api_key: OpenAI API key for the model.
        model: Model to use for the experiments.
        story: Name of the story to use.
        storydir: Directory containing the stories.
    Returns:
        Argument string.
    """
    return COMMON_ARGS.render(
        open_ai_api_key=open_ai_api_key if open_ai_api_key else '""',
        model=model,
        story=story_path(storydir=storydir, story=story),
    )


def create_output_dir(model: str,
                      modular_experiment_outputs: str,
                      stage: str,
                      setting_number: int,
                      iter_number: int,
                      premade_params_language: Optional[str] = None) -> str:
    """Creates the output directory for the experiment.

    Args:
        model: Model to use for the experiments.
        modular_experiment_outputs: Subdirectory for the experimental data.
        stage: Which morphosyntactic stage we are at.
        setting_number: Which setting number we are at.
        iter_number: Which iteration number we are at.
        premade_params_language: Language for the premade parameters.
    Returns:
        Path to created output directory.
    """
    base_dir = os.path.join(
        modular_experiment_outputs,
        model,
        premade_params_language if premade_params_language else "tmp",
    )
    path = os.path.join(
        base_dir,
        f"{stage}_{setting_number}_{iter_number}",
    )
    os.makedirs(path, exist_ok=True)
    return path


def dump_params(output_dir: str,
                params: Dict[str, str],
                original_story: str,
                previous_story: str) -> Dict[str, str]:
    """Dumps the params to the output directory.

    Args:
        output_dir: output directory.
        params: stage parameters.
        original_story: Path to the original story.
        previous_story: Path to the previous story.
    Returns:
        Same set of parameters.
    """
    with open(os.path.join(output_dir, "params.txt"), "w") as stream:
        for k in params:
            stream.write(f"{k}:\t{params[k]}\n")
            stream.write(f"Original story:\t{original_story}\n")
            stream.write(f"Previous story:\t{previous_story}\n")
    return params


def create_script(directory: str,
                  cmd: str,
                  script_name="script.sh") -> str:
    """Creates an executable script in the target directory.

    Args:
        directory: Path to output directory.
        cmd: Command string.
        script_name: Name for the script.
    Returns:
        Path to script.
    """
    script = os.path.join(directory, script_name)
    with open(script, "w") as stream:
        stream.write("#!/bin/bash\n")
        stream.write(f"{cmd}\n")
    os.chmod(script, 0o0755) # permission in octal
    # owner can read, write, and execute the file.
    # group and others can read and execute the file, but not write.

    return script


def run_word_order_experiment(story: str,
                              storydir: str,
                              model: str,
                              num_iter: int,
                              metascript_dir: str,
                              modular_experiment_outputs: str,
                              premade_params_language: str,
                              open_ai_api_key: Optional[str],
                              run_commands: bool = False) -> Dict[str, Any]:
    """First stage: Run the word order experiments.

    Args:
        story: Name of the story to use.
        storydir: Directory containing the stories.
        model: Model to use for the experiments.
        num_iter: Number of iterations for each setting.
        metascript_dir: Directory to store the metascripts.
        modular_experiment_outputs: Subdirectory for the experimental data.
        open_ai_api_key: OpenAI API key for the model.
        run_commands: Whether to actually run the commands.
    Returns:
        dict: Parameters for the experiment.
    """
    common_args = create_common_args(model=model,
                                     story=story,
                                     storydir=storydir,
                                     open_ai_api_key=open_ai_api_key)
    stage = "word_order"

    try:
        params = LANGUAGE_TO_PARAMS[premade_params_language]().model_dump()
    except KeyError:
        raise ValueError(f"Unsupported language: {premade_params_language}. "
                         "Supported languages are: "
                         f"{', '.join(LANGUAGE_TO_PARAMS.keys())}.")

    if stage == "word_order":
        stage_params = params["syntax"]
    # stage_params = which_params[stage]
    user_prompt = f"prompts/cumulative_morphosyntax/{stage}.txt"
    word_order_params = {}
    scripts = []

    # we are not setting multiple settings, so stage_params is a single-element list
    if not isinstance(stage_params, list):
        stage_params = [stage_params]
    for setting_number, features in enumerate(stage_params):
        # `params` are already model_dump()-ed; of type dict
        main_word_order = features["main_word_order"]
        oblique_word_order = features["oblique_word_order"]
        adj_noun_word_order = features["adj_noun_word_order"]
        posspron_noun_word_order = features["posspron_noun_word_order"]
        num_noun_word_order = features["num_noun_word_order"]
        adposition_noun_word_order = features["adposition_noun_word_order"]

        for it in range(num_iter):
            output_dir = create_output_dir(model=model,
                                           modular_experiment_outputs=modular_experiment_outputs,
                                           stage=stage,
                                           setting_number=setting_number,
                                           iter_number=it,
                                           premade_params_language=premade_params_language,)
            output = story_path(story=story,
                                storydir=output_dir)
            output_full = story_path(story=f"{story}_full",
                                     storydir=output_dir)
            user_prompt_dump = f"{output_dir}/user_prompt.txt"
            params = dump_params(output_dir=output_dir,
                                 params=features,
                                 original_story=story_path(storydir, story),
                                 previous_story="")
            cmd = " \\\n".join(
                [
                CREATE,
                common_args,
                f'    --output="{output}"',
                f'    --output_full="{output_full}"',
                f'    --user_prompt="{user_prompt}"',
                f'    --user_prompt_dump="{user_prompt_dump}"',
                '    --previous_translation=""',
                f'    --main_word_order="{main_word_order}"',
                f'    --oblique_word_order="{oblique_word_order}"',
                f'    --adj_noun_word_order="{adj_noun_word_order}"',
                f'    --posspron_noun_word_order="{posspron_noun_word_order}"',
                f'    --num_noun_word_order="{num_noun_word_order}"',
                f'    --adposition_noun_word_order="{adposition_noun_word_order}"',
                ]
            )
            if run_commands:
                os.system(cmd)
            word_order_params[output_dir] = {"params": params, "output": output}
            scripts.append(create_script(output_dir, cmd))
    cmd = "\n".join(scripts)
    create_script(metascript_dir,
                  cmd,
                  "word_order.sh")

    return word_order_params


def get_flag(stage: str) -> str:
    """Look up the relevant flag for this stage:

    Args:
        stage: Morphosyntactic stage.
    Returns:
        Flag relevant for this stage.
    """
    assert stage in STAGE_FLAGS
    return STAGE_FLAGS[stage]


def run_stage(stage: str,
              story: str,
              model: str,
              new_directory: str,
              previous_translation: str,
              params: Dict[str, Any],
              previous_params: Dict[str, Any],
              storydir: str,
              premade_params_language: str,
              open_ai_api_key: str | None = None,
              run_commands: bool = False) -> Tuple[Dict[str, Any], str, str]:
    """Runs an individual stage in the grammar construction.

    Args:
        stage: Which morphosyntactic stage we are at.
        story: Name of the story to use.
        model: Model to use for the experiments.
        new_directory: Directory for this stage on this iteration.
        previous_translation: Path to previous translation.
        params: Params specific to this stage.
        previous_params: Cumulated parameters up to now.
        storydir: Directory containing the stories.
        premade_params_language: Language name of the params used here. Necessary for choosing the correct review prompt file.
        open_ai_api_key: OpenAI API key for the model.
        run_commands: Whether to actually run the commands.
    Returns:
        Tuple of parameters up to this stage, the path to the
        output for the next phase, and the script produced.
    """
    previous_params = copy.deepcopy(previous_params)
    os.makedirs(new_directory, exist_ok=True)
    common_args = create_common_args(open_ai_api_key=open_ai_api_key,
                                     model=model,
                                     story=story,
                                     storydir=storydir)
    output = story_path(story=story,
                        storydir=new_directory)
    output_full = story_path(story=f"{story}_full",
                             storydir=new_directory)
    
    if stage == "review":
        user_prompt = f"prompts/cumulative_morphosyntax/{stage}/{stage}_{premade_params_language}.txt"
        params["review"] = True # Placeholder
        # When it is the review stage, we need to pass all of the parameters to create.py.
    else:
        user_prompt = f"prompts/cumulative_morphosyntax/{stage}.txt"
    user_prompt_dump = f"{new_directory}/user_prompt.txt"
    stage_params = params[stage] # this can also be another BaseModel
    # if stage_params is BaseModel, it has to be passed to `cmd` later as a a string.
    # for example, if it looks like `stage_params = Sample(category='example', value=1)`, then
    # str(stage_params) will give us 'category=example value=1'.
    # then, later in create.py, we will have to reconstruct the original dict in the cumulative_morphosyntax_params(),
    # which maps the cmd flag name to the actual parameter and will be called in create.py().

    print(f"{stage}: {stage_params}")  # debug

    previous_params[stage] = stage_params
    previous_params = dump_params(
        new_directory,
        previous_params,
        story_path(storydir, story),
        previous_translation,
    )

    flag = get_flag(stage)
    cmd = [
        CREATE,
        common_args,
        f'   --output="{output}"',
        f'   --output_full="{output_full}"',
        f'   --user_prompt="{user_prompt}"',
        f'   --user_prompt_dump="{user_prompt_dump}"',
        f'   --previous_translation="{previous_translation}"',
    ]
    if flag:
        if stage == "review":
            # Pass all the morphosyntactic parameters.
            for stg, param in params.items():
                print(stg, param)
                # key is the stage name, param is the parameter value
                try:
                    flag = get_flag(stg)
                    if flag is None or str(param) == "None":
                        print(f"Skipping the review stage for stage {stg} because either the flag or the parm is None.")
                        continue
                    cmd.append(f'    --{flag}="{str(param)}"')
                except:
                    print(f"Skipping the review stage for stage {stg} because it is not defined in STAGE_FLAGS.")
                    continue
        else:
            cmd.append(f'    --{flag}="{str(stage_params)}"')  # This is typically added for morphological stages.
    cmd = " \\\n".join(cmd)
    if run_commands:
        os.system(cmd)
    script = create_script(new_directory, cmd)
    return previous_params, output, script


def run_rest(story: str,
             storydir: str,
             model: str,
             word_order_params: Dict[str, Any],
             metascript_dir: str,
             premade_params_language: str,
             open_ai_api_key: Optional[str] = None,
             run_commands: bool = False,
             do_review: bool = False) -> None:
    """Run the rest of the morphosyntactic construction.

    Args:
        story: Name of the story to use.
        storydir: Directory containing the stories.
        model: Model to use for the experiments.
        word_order_params: Parameters from the previous word-order run.
        metascript_dir: Directory to store the metascripts.
        open_ai_api_key: OpenAI API key for the model.
        run_commands: Whether to actually run the commands.
        do_review: Whether to run the review stage (few-shot ICL).
    """
    try:
        params = LANGUAGE_TO_PARAMS[premade_params_language]().model_dump()
    except KeyError:
        raise ValueError(f"Unsupported language: {premade_params_language}. "
                         "Supported languages are: "
                         f"{', '.join(LANGUAGE_TO_PARAMS.keys())}.")
    print(params)

    # we are not setting multiple settings, so stage_params is a single-element list
    inflection_params = params["morphology"]
    if not isinstance(inflection_params, list):
        inflection_params = [inflection_params]

    for directory in word_order_params:
        # for i, features in enumerate(params["rest"]):
        for i, features in enumerate(inflection_params):
            previous_translation = word_order_params[directory]["output"]
            previous_params = word_order_params[directory]["params"]
            scripts = []
            for stage in STAGES:
                # print(features)
                if features.get(stage) is None:
                    if do_review and stage == "review":
                        print("review stage found.")
                    else:
                        print(f"Skipping stage {stage} as it has no parameters.")
                        continue
                previous_params, previous_translation, script = run_stage(
                    stage=stage,
                    story=story,
                    model=model,
                    new_directory=new_directory(directory, stage, i),
                    previous_translation=previous_translation,
                    params=features,
                    previous_params=previous_params,
                    storydir=storydir,
                    premade_params_language=premade_params_language,
                    open_ai_api_key=open_ai_api_key,
                    run_commands=run_commands
                )
                scripts.append(script)
            cmd = "\n".join(scripts)
            script_name = new_directory(directory=directory,
                                        stage="morphosyntax",
                                        idx=i)
            script_name = script_name.split("/")[-1] + ".sh"
            create_script(metascript_dir, cmd, script_name)


def run_evaluation(output_dir: str,
                   model: str,
                   reference_file: str,
                   metascript_dir: str,
                   premade_params_language: str,
                   num_iter: int,
                   prediction_file_name: str = "grammatical_test_sentences.txt") -> None:
    """Create the evaluation scripts."""
    # TODO: add a subdirectory for each language.

    # look for the last stage directories
    script_dir = os.path.join(
        output_dir,
        model,
        premade_params_language if premade_params_language else "tmp",
    )
    script_files = glob.glob(script_dir + "/*")

    last_stage = find_last_stage(stages=STAGES,
                                 output_files=script_files)
    if last_stage is None:
        raise ValueError("No stages found in the scripts directory.")
    print(f"Last stage found: {last_stage}")
    print(f"Creating evaluation scripts based on the last output in {last_stage}.")

    for i in range(num_iter):
        prediction_file = os.path.join(script_dir, f"{last_stage}_0_{i}_0", prediction_file_name)
        # In evaluation/eval_morphosyntax.py, add a step to structuralize the output text for output format standardization.
        # Then, based on the JSON structured data, run the evaluation.
        results_file = os.path.join(script_dir, f"structured_result_0_{i}_0.csv")
        scores_file = os.path.join("evaluation",
                                   "results",
                                   model,
                                   premade_params_language,
                                   f"scores_0_{i}.json")
        if not os.path.exists(os.path.dirname(scores_file)):
            os.makedirs(os.path.dirname(scores_file), exist_ok=True)
        scripts = [
            "python3 evaluation/eval_morphosyntax.py",
            f"    --model_outputs_file {prediction_file}",
            f"    --reference_glosses {reference_file}",
            "    --pipeline",
            f"    --results_file {results_file}",
            f"    --scores_file {scores_file}",
        ]
        cmd = " \\\n".join(scripts)
        script_name = f"evaluation_0_{i}.sh"
        create_script(metascript_dir, cmd, script_name)


def new_directory(directory: str,
                  stage: str,
                  idx: str | int):
    """Creates a new directory for the given stage and iteration.

    """
    directory = directory.replace("word_order", stage)
    return f"{directory}_{idx}"


def main(args: argparse.Namespace) -> None:
    """Main function."""
    # test
    if args.pipeline:
        assert args.reference_file, "--reference_file is required when --pipeline is set."

    metascript_dir = os.path.join(
        args.modular_experiment_outputs,
        args.model,
        args.premade_params_language if args.premade_params_language else "tmp",
        "metascripts",
    )
    os.makedirs(metascript_dir, exist_ok=True)

    # First, tweak the syntax (word order) parameters.
    word_order_params: dict = run_word_order_experiment(
        story=args.story,
        storydir=args.storydir,
        model=args.model,
        num_iter=args.num_iter,
        metascript_dir=metascript_dir,
        modular_experiment_outputs=args.modular_experiment_outputs,
        open_ai_api_key=args.open_ai_api_key,
        run_commands=args.run_commands,
        premade_params_language=args.premade_params_language,
    )

    # Then, tweak the morphology parameters.
    run_rest(
        story=args.story,
        storydir=args.storydir,
        model=args.model,
        word_order_params=word_order_params,
        metascript_dir=metascript_dir,
        premade_params_language=args.premade_params_language,
        do_review=args.do_review,
    )

    run_evaluation(
        output_dir=args.modular_experiment_outputs,
        model=args.model,
        reference_file=args.reference_file,
        metascript_dir=metascript_dir,
        premade_params_language=args.premade_params_language,
        num_iter=args.num_iter,
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
