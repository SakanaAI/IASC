<img src="https://github.com/SakanaAI/IASC/blob/main/iasc.png" alt="iasc" width="200"/>

# IASC: Interactive Agentic System for ConLangs

The term "Constructed Language"---often shortened to "ConLang"---is used to
refer to any artificially created language that is intended, in principle, to be
as expressive as naturally evolved human languages. The latter restriction is
important since ConLangs are to be distinguished from artificial languages, such
as mathematical symbology, which are artificial languages (at least in the
formal sense of language) to be sure, but which are much more limited in the
kinds of messages they can convey.

When people think of ConLangs, the first things that come to mind may be
languages like Esperanto or Interlingua, which are artificial languages designed
to be like natural languages, though with fewer of the grammaical complexities
one often finds in real natural languages. ConLangs in this category were
intended by their creators to serve as substitutes for naturally evolved
languages.

Or, second, one might think of fantasy languages, such as the languages of
Tolkien’s Middle Earth---Elvish (Quenya) or the language of Mordor. Again, these
are artificial languages that nonetheless bear a strong resemblance to natural
languages. Quenya, for example, looks very Finno-Ugri: see
[https://en.wikipedia.org/wiki/Finnish_influences_on_Tolkien](https://en.wikipedia.org/wiki/Finnish_influences_on_Tolkien).

A third category are invented alien languages, such as Klingon or Vulcan from
the Star Trek franchise which, at least in principle, were designed to be
somewhat unlike typical human languages in their construction.

Finally, there is a fourth category of languages that are supposed to be based
on "logical" principles, such as LogLan or John Wilkins’ "philosophical
language".

In this research, we are interested in discovering to what degree Large Language Models can help
in the creation of ConLangs that are much like real human languages in their properties, but are not
intended to replace naturally evolved languages. Thus the ConLangs created basically fit into the
second notion of sketched above.

# Preliminaries: Requirements and LLM access

First of all:

```
pip3 install -r requirements.txt
```

For Anthropic (Claude series) models we assume access via AWS.  Make sure that
your environment recognizes AWS environment variables.  To check this, run `echo
$AWS_ACCESS_KEY_ID` and `echo $AWS_SECRET_ACCESS_KEY`, and make sure some value
is returned. If not, set the environment variables `export
AWS_ACCESS_KEY_ID=<your access key id>` and `export AWS_SECRET_ACCESS_KEY=<your
secret acces key>`.  If you are using conda (which we use), also consider adding
the environment variables in the virtual environment by `conda env config vars
set AWS_ACCESS_KEY_ID=<your access key id>` etc.

For OpenAI and Gemini you will need to pass your API key as a command-line argument.

# Getting Started

A good place to start is with the scripts named `controlled_*_claude*.sh` in the
subdirectories `modular_experiments` and `modular_experiment/controlled_morphosyntax`.

The basic protocol is to:

- Generate a phonology (phonotactics) for the language
- Generate the morphosyntax
- Generate an initial lexicon
- Generate the orthography
- Generate an updated lexicon with spellings
- Generate a handbook --- a short grammar describing the language.
- Optionally, translate more texts into the target language.

Examples for various language configurations are given below:

## Run everything (example)
- Japanese-like phonology
- Turkish-like morphosyntax
```
. ./modular_experiments/controlled_phonology_claude.sh
. ./modular_experiments/controlled_morphosyntax/claude/turkish.sh
. ./modular_experiment_outputs_controlled/claude/turkish/metascripts/word_order.sh
. ./modular_experiment_outputs_controlled/claude/turkish/metascripts/morphosyntax_0_0_0.sh
. ./modular_experiment_outputs_controlled/claude/turkish/metascripts/morphosyntax_0_1_0.sh
. ./modular_experiment_outputs_controlled/claude/turkish/metascripts/morphosyntax_0_2_0.sh
. ./modular_experiment_outputs_controlled/claude/french/metascripts/evaluation_0_0.sh
. ./modular_experiment_outputs_controlled/claude/french/metascripts/evaluation_0_1.sh
. ./modular_experiment_outputs_controlled/claude/french/metascripts/evaluation_0_2.sh
. ./modular_experiments/controlled_corpus_creation_claude_0.sh
. ./modular_experiments/controlled_orthography_claude.sh
. ./modular_experiments/controlled_corpus_creation_claude_1.sh
. ./modular_experiments/controlled_handbook_claude.sh
```

## Run (morphosyntax)
This workflow does not include instructions on phonotactics/phonology.
The example below uses Claude as the LLM and the French-like feature sets.

- `./modular_experiments/controlled_morphosyntax/claude/french.sh`
- `./modular_experiment_outputs_controlled/claude/french/metascripts/word_order.sh`
- `./modular_experiment_outputs_controlled/claude/french/metascripts/morphosyntax_0_0_0.sh`
- `./modular_experiment_outputs_controlled/claude/french/metascripts/morphosyntax_0_1_0.sh`
- `./modular_experiment_outputs_controlled/claude/french/metascripts/morphosyntax_0_2_0.sh`

A metascript `morphosyntax_0_0_0.sh` includes the following shell scripts (for now):
- `./modular_experiment_outputs_controlled/claude/inclusive_exclusive_0_0_0/script.sh`
- `./modular_experiment_outputs_controlled/claude/number_0_0_0/script.sh`
- `./modular_experiment_outputs_controlled/claude/case_marking_0_0_0/script.sh`
- `./modular_experiment_outputs_controlled/claude/tense_aspect_0_0_0/script.sh`
- `./modular_experiment_outputs_controlled/claude/person_agreement_0_0_0/script.sh`
- `./modular_experiment_outputs_controlled/claude/extras_0_0_0/script.sh`

To run an evaluation,
- `./modular_experiment_outputs_controlled/claude/french/metascripts/evaluation_0_0.sh`

## The outputs
- After you run `modular_experiment_outputs_controlled/claude/metascripts/word_order.sh`, you get the
  intermediate output in `modular_experiment_outputs_controlled/claude/word_order_0_{j}/sentence_design_output.txt`.
- The final output can be found in `modular_experiment_outputs_controlled/claude/`

## To run an experiment for all languages
You can run an experiment across all the languages, including the evaluation step, by running the following command:
- `./modular_experiments/controlled_morphosyntax/claude/all_languages.sh`

## To add a new morphosyntactic feature in the Morphosyntax pipeline
If you add a new morphological feature stage, then you should see the corresponding folder (like `number_0_0_0/`) here.

and each shell script, for example `inclusive_exclusive_0_0_0/script.sh`, looks like this:

```
#!/bin/bash
python3 create/create.py \
    --model="claude" \
    --task="cumulative_morphosyntax" \
    --sample_text="sentence_design_output/grammatical_test_sentences.txt" \
    --open_ai_api_key="" \
    --num_iterations=1 \
    --output="modular_experiment_outputs_controlled/claude/inclusive_exclusive_0_0_0/sentence_design_output.txt" \
    --output_full="modular_experiment_outputs_controlled/claude/inclusive_exclusive_0_0_0/sentence_design_output_full.txt" \
    --user_prompt="prompts/cumulative_morphosyntax/inclusive_exclusive.txt" \
    --user_prompt_dump="modular_experiment_outputs_controlled/claude/inclusive_exclusive_0_0_0/user_prompt.txt" \
    --previous_translation="modular_experiment_outputs_controlled/claude/word_order_0_0/sentence_design_output.txt" \
    --inclusive_exclusive="False"
```

To run this correctly, make sure to have the `user_prompt` ready.

When you run `create/create.py` with the `cumulative_morphosyntax` task,
- it will call `create/create_lib.py`'s `run_model(client, params, output_path, output_path_full, user_prompt_path, modular_morphosyntax, user_prompt_dump=user_prompt_dump,` function.
- `run_model()` function first calls `utils/common_utils.py`'s `create_user_prompt(params, user_prompt_path, modular_morphosyntax)` function.
  - This function gets the corresponding prompt text file first, and calls the `modular_morphosyntax_prompt()` function to prepare the prompt text via `load_system_instruction()` function. The comments `<!-- ... -->` and the instruction tags `<INSTRUCTIONS>` `</INSTRUCTIONS>` are removed.
  - The loaded user prompt contains placeholders for specific morphosyntactic parameters. These placeholders are filled out by the passed morphosyntactic parameters via `jinja`'s `Template(prompt).render(**params)`.
  - So, make sure that the keys in `params` (defined in `morphosyntax_params.py`) correctly corresponds to the placeholder variable names in the user prompt.

So, if you want to add a new morphological feature, below are the points for modification:
- `prompts/cumulative_morphosyntax/`: In this folder, make a new prompt file for the newly added feature.
- Add the new feature to either `Syntax` class or `Morphology` class in `modular_experiments/morphosyntax_params.py`. Also update the sample parameter sets if necessary.
  - You may also add a nested BaseModel value to a key. (only one-level nest is supported for now.)
  - If the value is BaseModel, the corresponding flag value in a command should be in string. (see `RELATIVIZATION` defined in `common_utils.py` for an example.)
  - Then, the passed string is re-converted to a dictionary in `cumulative_morphosyntax_params()` in `common_utils.py`, which is loaded in `run_morphosyntax.py()`.
- In `modular_experiments/run_morphosyntax.py`,
  - If you are adding a syntactic feature, add to the `run_word_order_experiment()` the following:
    - Add a line to load the newly added feature (e.g. `main_word_order = features["main_word_order"]`)
    - Add an argument to a command line string to pass the newly added feature (e.g. `f'   --adj_noun_word_order="{adj_noun_word_order}"'`).
  - If you are adding a morphological feature,
    - Add the stage (i.e., the generation step to process the morphological feature) in the parameter `STAGE`.
    - Add it to the dictionary `STAGE_FLAGS`. The key should be the stage name in `STAGE`, and the value should be the root name of the prompt file (e.g., if the prompt file name is `stage.txt`, the value should be `stage`.)
    - If your new morphological feature is a nested BaseModel, then make sure to reconstruct the original dict structure in `cumulative_morphosyntax_params()`.
- In `utils/common_utils.py`,
  - Add a command line argument variable for the newly added feature with `flags.DEFINE_string` or `flags.DEFINE_list` (an absl feature)
  - Add the new feature and its corresponding file name to `feature_to_file`. It is safe to have the same names in a key and its value.
  - Add the new feature to the return values of `cumulative_morphosyntax_params()` to map the flag value to the params dictionary. This will be called in `create.py`.
- In `modular_experiments/run_handbook.py`, update the features passed to `cmd` in `run_handbook_experiment()`.

If you add a feature that is not a module (stage) but is processed by an existing module, then you need to:
- define a feature in `modular_experiments/run_morphosyntax.py`;
- add a placeholder in the corresponding existing prompt file.

But:
- You do not have to create a new prompt file;
- You do not have to add a stage to `STAGE` in `modular_experiments/run_morphosyntax.py`.


## If you want to add a new language model to run
First, go to `llm/llm.py` and add the new LLM's name to the flag options in `MODEL`.
If the model needs an API key, make sure you have the API key set as an environment variable.
To check whether you have set it ready in the environment variables, you can run `conda env config vars list`.
To add, `conda env config vars set <API_KEY_NAME>=<your_api_key>`.

Then, add a code block to instantiate an LLM client for the newly added model in the `client()` function of `llm/llm.py`.
If it is a new variant of an already added model family (e.g., another Gemini model), then make sure that the existing condition will correctly load the newly added model.

Next, add a code block to `llm_predict()` in `llm/llm.py` to run the LLM inference through API.

Finally, add the newly added model name to the `--model` argument in the argument parser in `get_args()` in `modular_experiments/run_morphosyntax.py`.

## Run translation
`modular_experiments/controlled_translation_claude.sh` gives an illustration of how to translate a new text into the target language. This can then
be followed by `modular_experiments/controlled_corpus_creation_claude_3.sh`, which will add new words to the lexicon and produce a written
corpus of the new translation using the language's orthography. This assumes you
have created a language with Japanese-like phonology, Arabic-like morphosyntax, and
which uses the Latin script.

## FAQ
- `ERROR: Can't invoke 'us.anthropic.claude-3-5-sonnet-20240620-v1:0'. Reason: Unable to locate credentials.`:
    This happens when the AWS Bedrock cannot find your AWS credentials in the environment variables.
    Make sure you have your AWS credentials and have set them as the environmental variables.
- `**ERROR** in sample text in the user prompt: the sample text depends on the earlier morphosyntax outputs by Claude.
    If any of the preceding tasks did not run successfully, you do not get the output.
    Make sure that the preceding tasks are successfully executed.


# Evaluation data
The structure of the evaluation is as follows:
- Input:
  - input text (str): The input text (English) to translate.
  - features (dict-like): The morphosyntactic features that our desired conlang has.
- Output:
  - output text (str): The output text as a gloss.

Let's say that the input text is "He is the smartest student in our school." and the input features are the French-like set.
Then, the output should be something like "he be-PRES-3SG the student-SING smart-SUP in we-GEN school."

<!--
An example French-like input feature set looks like this:
```
def sample_params_french():
    """Sample parameters like French."""
    return Morphosyntax(
        general_morphology=GeneralMorphology(
            main_word_order="SVO",
            adj_noun_word_order="NA",
            adposition_noun_word_order="PN",
            morphology_type="fusional",
            alignment="nominative-accusative"
        ),
        inflection=Inflection(
            case_marking=None,
            gender=["masculine", "feminine"],
            definiteness=["definite", "indefinite"],
            tense_aspect=["present", "past", "future", "imperfective"],
            mood=["indicative", "subjunctive", "imperative", "conditional"],
            person_agreement=["first", "second", "third"],
            voice=["active", "passive"],
            inclusive_exclusive=False,
            number=["singular", "plural"]
        )
    )
```
-->

# Publication

Chihiro Taguchi and Richard Sproat. 2025. ``IASC: Interactive Agentic System for ConLangs''. arXiv.
