"""Run phonotactics and phonological rules, and check output."""

import agentic_phonology.loader as loader
import agentic_phonology.prompter as prompter
import jsonlines
import random
import re

from absl import flags
from absl import logging

from llm import llm
from phonology.phoible import PHOIBLE
from typing import Any, Dict, Sequence, Tuple


# Special language "Color" gets the Color language.
LANGUAGE = flags.DEFINE_string(
  "language",
  None,
  "Language to model phonotactics on.",
)
PHONOTACTICS_BASE = flags.DEFINE_string(
  "phonotactics_base",
  None,
  "Path to base name for phonotactics.",
)
PHONRULES_BASE = flags.DEFINE_string(
  "phonrules_base",
  None,
  "Path to base name for phonological rules.",
)
MAX_ITER = flags.DEFINE_integer(
  "max_iter",
  10,
  "Maximum allowed iterations.",
)
NUM_PHONOTACTICS_EXAMPLES = flags.DEFINE_integer(
  "num_phonotactics_examples",
  100,
  "Number of phonotactics examples",
)
NUM_OUTPUT_EXAMPLES = flags.DEFINE_integer(
  "num_output_examples",
  20,
  "Number of examples of input-output pairs for phonological rules",
)
NUM_CLOSEST = flags.DEFINE_integer(
  "num_closest",
  1,
  "Number of closest historical languages to pick.",
)
USER_PROMPT_DUMP = flags.DEFINE_bool(
  "user_prompt_dump",
  True,
  "Dump the user prompt to a file.",
)


def run_phonotactics(
    input_phonotactics: str,
    stress_placement: str="penultimate",
    num_examples: int=100,
    add_word_boundaries: bool=False,
) -> Sequence[str]:
  params = loader.load_phonotactics(input_phonotactics, stress_placement)

  def generate_morpheme():
    output = params["morpheme_generator"]().strip()
    if add_word_boundaries:
      output = f"# {output} #"
    return output

  return [generate_morpheme() for _ in range(num_examples)]


COMMENT = re.compile(r"^#+")


def check_changes_claimed_for_rules(
    phonrule_params: Tuple[str, Any],
):
  """Check the individual rules against what the system claims they should do.

  Look at the comments the LLM created on the last round as examples of what
  each individual rule should do. Then apply each rule, checking to see that the
  output corresponds to the input.
  """
  rule_def = False
  examples = []
  for line in phonrule_params["source_code"].split("\n"):
    if line.startswith("def "):
      rule_def = True
      rule_name = line.split("def ")[-1].split("(")[0].strip()
      continue
    if rule_def:
      line = line.strip()
      if line.startswith("#"):
        line = COMMENT.sub("", line)
        if "Example inputs" in line:
          continue
        try:
          inp, out = line.split("->")
          inp = inp.strip()
          out = out.strip()
        except ValueError:
          inp = "??"
          out = line.strip()
        examples.append((rule_name, inp, out))
      else:
        rule_def = False
  errors = []
  for rule_name, inp, out in examples:
    if inp == "??":
      msg = (
        f"Your example input/output specification `{out}` is "
        f"ill-formed for rule `{rule_name}`"
      )
      errors.append(msg)
      continue
    try:
      # TODO(rws): Again, this is potentially dangerous:
      rule = eval(f'phonrule_params["module"].{rule_name}')
    except KeyError:
      errors.append(f"Rule `{rule_name}` is missing")
      continue
    try:
      actual = rule(inp)
      if actual != out:
        msg = (
          f"Your rule `{rule_name}` applied to `{inp}` does not match the "
          f"expected output:\nExpected = `{out}`; Actual = `{actual}`\n"
        )
        errors.append(msg)
    except Exception as e:
      msg = (
        f"Your rule `{rule_name}`, applied to `{inp}` raises an Exception: {e}"
      )
      errors.append(msg)
  return "\n".join(errors)


def run_phonrules(
    phonrule_params: Tuple[str, Any],
    input_phonotactics: str,
    stress_placement: str="penultimate",
) -> Sequence[Tuple[str, str]]:
  inputs = run_phonotactics(
    input_phonotactics,
    stress_placement,
    num_examples=1_000,
    add_word_boundaries=True,
  )
  changes = []
  for inp in inputs:
    out = phonrule_params["rules"](inp)
    if inp != out:
      changes.append((inp, out))
  return changes


MULTI_SPACE = re.compile(r"\s\s+")

# Valid phones include "." for syllable boundary, "#" for word boundary, "-" for
# morpheme boundary, plus some that are oddly missing from Phoible:
VALID_PHONES = [".", "#", "-", "tʃ", "dʒ"]


def score_spacing(transcription: str) -> Dict[str, Any]:
  """Checks for spacing issues.
  """
  suspect_phones = set()
  for phone in transcription.split():
    if len(phone) == 1:
      continue
    if phone not in PHOIBLE.all_phonemes + VALID_PHONES:
      suspect_phones.add(phone)
  multi_boundary = False
  if "# #" in transcription:
    multi_boundary = True
  return {
    "transcription": transcription,
    "suspect_phones": suspect_phones,
    "multi_space": len(MULTI_SPACE.findall(transcription)),
    "multi_boundary": multi_boundary,
  }


# Loads phonesets from precompiled jsonl from Diachronica.


DIACHRONICA = "agentic_phonology/diachronica/diachronica_expanded_2.jsonl"


def load_diachronica_phonesets() -> Dict[str, Any]:
  table = {}

  def compute_features(phonemes):
    result = []
    for ph in phonemes:
      try:
        features = PHOIBLE.ph_features(ph)
      except KeyError:
        features = None
      result.append(features)
    return result

  with jsonlines.open(DIACHRONICA) as reader:
    for elt in reader:
      language = elt["daughter"]
      phonemes = elt["daughter_phonemes"]
      rule_sets = elt["daughter_rule_sets"]
      features = compute_features(phonemes)
      table[language] = {
        "phonemes": phonemes,
        "features": features,
        "rule_sets": rule_sets,
      }
      # Repeat for parent if not there already:
      parent = elt["parent"]
      if parent in table:
        continue
      phonemes = elt["parent_phonemes"]
      name = f"{parent} to {language}"
      rule_sets = [[name, elt["parent_to_daughter_rules"]]]
      features = compute_features(phonemes)
      table[parent] = {
        "phonemes": phonemes,
        "features": features,
        "rule_sets": rule_sets,
      }
  return table


DIACHRONICA_PHONESETS = load_diachronica_phonesets()


def closest_phoneset(
    phoneset: Sequence[str],
) -> Sequence[Dict[str, Any]]:
  results = []
  for language in DIACHRONICA_PHONESETS:
    lphoneset = DIACHRONICA_PHONESETS[language]["phonemes"]
    features = DIACHRONICA_PHONESETS[language]["features"]
    closest = []
    dist = 0
    for feature_set in features:
      ph, d = PHOIBLE.closest_phoneme(feature_set, subset=phoneset)
      closest.append(ph)
      dist += d
    dist /= len(lphoneset)  # Normalize by length of target phonemes
    # Add penalty for length differences
    ratio = len(set(closest)) / len(phoneset)
    if ratio < 1:
      ratio = 1 / ratio
    dist += ratio
    results.append(
      {
        "distance": dist,
        "phoneset": lphoneset,
        "language": language,
        "rule_sets": DIACHRONICA_PHONESETS[language]["rule_sets"],
      }
    )
  results.sort(key=lambda x: x["distance"])
  return results


def build_phonotactics_prompt(language: str) -> str:
  if LANGUAGE.value.lower() == "color":
    return prompter.PHOTOTACTICS_BASIC_INSTRUCTIONS.render()
  else:
    return prompter.PHONOTACTICS_BASIC_INSTRUCTIONS.render(
      language=language.replace("_", " "),
    )


def build_phonotactics_critique_prompt(
    phonotactics: str,
    phonotactics_basic_prompt: str,
    stress_placement: str="penultimate",
    num_output_examples: int=20,
    which_attempt: int=1,
) -> str:
  phonotactics_basic_prompt = (
    "The following are the instructions you were originally given:\n\n" +
    "<ORIGINAL_INSTRUCTIONS>\n" +
    phonotactics_basic_prompt +
    "\n\n" +
    "</ORIGINAL_INSTRUCTIONS>\n"
  )
  params = loader.load_phonotactics(phonotactics)
  outputs_from_last_round = ""
  try:
    morphemes = run_phonotactics(
      input_phonotactics=phonotactics,
      stress_placement=stress_placement,
      add_word_boundaries=False,
      num_examples=1_000,
    )
  except Exception as e:
    morphemes = []
    assessment = (
      f"Your phonotactics code raises an Exception: {e}"
    )
    outputs_from_last_round = "No outputs from last round due to an error."
  if morphemes:
    scores = [score_spacing(m) for m in morphemes]
    multi_space = set()
    suspect_phones = set()
    multi_boundary = []
    for k in scores:
      suspect_phones = suspect_phones.union(k["suspect_phones"])
      if k["multi_space"]:
        multi_space.add(k["transcription"])
      if k["multi_boundary"]:
        multi_boundary.append(k["transcription"])
    multi_space = list(multi_space)
    random.shuffle(multi_space)
    multi_space = multi_space[:3]
    random.shuffle(multi_boundary)
    multi_boundary = multi_boundary[:3]
    assessment = []
    num = 1
    if suspect_phones:
      assessment.append(
        f"{num}. The following combined phonemes suggest that your code is not "
        "correctly dealing with spaces between phones. Please correct this:\n"
      )
      assessment.append(", ".join(list(suspect_phones)))
      assessment.append("\n")
      num += 1
    if multi_space:
      assessment.append(
        f"{num}. The following examples contain more than one space between "
        "phones, suggesting that your code mistakenly adds extra spaces, "
        "or fails to clean up extra spaces:\n"
      )
      assessment.append(", ".join([f"/{m}/" for m in multi_space]))
      num += 1
    if multi_boundary:
      assessment.append(
        f"{num}. In the following you have duplicate word boundaries. "
        "Please fix:\n"
      )
      assessment.append(", ".join([f"/{m}/" for m in multi_boundary]))
    assessment = "\n".join(assessment).strip()
    random.shuffle(morphemes)
    outputs_from_last_round = "\n".join(morphemes[:num_output_examples])

  critique = ""
  if assessment:
    critique += (
      "The issues below seem to point to bugs in your previous code in one " +
      "or more of your rules:\n\n" +
      assessment +
      "\n\n" +
      "Please correct the errors noted above."
    )
  if LANGUAGE.value.lower() == "color":
    improvement_instructions = prompter.PHOTOTACTICS_IMPROVEMENT_INSTRUCTIONS
  else:
    improvement_instructions = prompter.PHONOTACTICS_IMPROVEMENT_INSTRUCTIONS
  prompt = prompter.build_whole_prompt(
    basic_instructions=phonotactics_basic_prompt,
    source_code=params["source_code"],
    outputs_from_last_round=outputs_from_last_round,
    critique=critique,
    improvement_instructions=improvement_instructions,
    which_task="phonotactics",
    which_attempt=which_attempt,
  )
  return prompt


def build_phonological_rules_prompt(
    phonotactics: str,
    nclosest: int=1,
    stress_placement: str="penultimate",
    num_examples: int=20,
) -> str:
  params = loader.load_phonotactics(phonotactics, stress_placement)
  consonants = params["consonants"]
  vowels = params["vowels"]
  try:
    sample_words = "\n".join(
      run_phonotactics(
        phonotactics,
        stress_placement,
        num_examples=num_examples,
        add_word_boundaries=True,
      )
    )
  except Exception as e:
    sample_words = f"Your phonotactics code raises an Exception: {e}"
  phoneset = [p.strip() for p in consonants.split(",") + vowels.split(",")]
  closest = closest_phoneset(phoneset)[:nclosest]
  tuples = []
  for table in closest:
    rule_sets = []
    for rule_set in table["rule_sets"]:
      rule_sets.append((rule_set[0], "\n".join(rule_set[1])))
    tuples.append((table["language"], ", ".join(table["phoneset"]), rule_sets))
  lines = []
  for language, phoneset, rulesets in tuples:
    lines.append("")
    lines.append(f"Language: {language}")
    lines.append(f"Phoneset: {phoneset}")
    for i, (name, ruleset) in enumerate(rulesets):
      lines.append("")
      lines.append(f"Ruleset {i + 1} ({name}):")
      lines.append(ruleset)
  reconstructions = "\n".join(lines)
  prompt = prompter.PHONOLOGICAL_RULES_BASIC_INSTRUCTIONS.render(
    consonants=consonants,
    vowels=vowels,
    num=nclosest,
    reconstructions=reconstructions,
    stress_placement=stress_placement,
    sample_words=sample_words,
  )
  prompt = prompter.build_whole_prompt(
    basic_instructions=prompt,
  )
  return prompt


def build_phonological_rules_critique_prompt(
    phonotactics: str,
    phonrules: str,
    phonological_rules_basic_prompt: str,
    stress_placement: str="penultimate",
    num_output_examples: int=20,
    which_attempt: int=1,
    code_unchanged: bool=False,
) -> str:
  phonological_rules_basic_prompt = (
    "The following are the instructions you were originally given:\n\n" +
    "<ORIGINAL_INSTRUCTIONS>\n" +
    phonological_rules_basic_prompt +
    "\n\n" +
    "</ORIGINAL_INSTRUCTIONS>\n"
  )
  phonrule_params = loader.load_phonrules(phonrules)
  try:
    pairs = run_phonrules(
      phonrule_params=phonrule_params,
      input_phonotactics=phonotactics,
      stress_placement=stress_placement,
    )
  except Exception as e:
    pairs = []
    assessment = [
      f"Your phonological rules code raises an Exception: {e}"
    ]
    outputs_from_last_round = "No outputs from last round due to an error."
  if pairs:
    scores = [score_spacing(p[1]) for p in pairs]
    multi_space = set()
    suspect_phones = set()
    multi_boundary = []
    for k in scores:
      suspect_phones = suspect_phones.union(k["suspect_phones"])
      if k["multi_space"]:
        multi_space.add(k["transcription"])
      if k["multi_boundary"]:
        multi_boundary.append(k["transcription"])
    multi_space = list(multi_space)
    random.shuffle(multi_space)
    multi_space = multi_space[:3]
    random.shuffle(multi_boundary)
    multi_boundary = multi_boundary[:3]
    assessment = []
    num = 1
    if suspect_phones:
      assessment.append(
        f"{num}. The following combined phonemes suggest that your code is not "
        "correctly dealing with spaces between phones, or that you are producing "
        "incorrect IPA output. Please correct this:\n"
      )
      assessment.append(", ".join(list(suspect_phones)))
      assessment.append("\n")
      num += 1
    if multi_space:
      assessment.append(
        f"{num}. The following examples contain more than one space between "
        "phones, suggesting that your code mistakenly adds extra spaces, "
        "or fails to clean up extra spaces:\n"
      )
      assessment.append(", ".join([f"/{m}/" for m in multi_space]))
      num += 1
    if multi_boundary:
      assessment.append(
        f"{num}. In the following you have duplicate word boundaries. "
        "Please fix:\n"
      )
      assessment.append(", ".join([f"/{m}/" for m in multi_boundary]))
    random.shuffle(pairs)
    outputs_from_last_round = "\n".join(
      [f"/{p[0]}/ -> /{p[1]}/" for p in pairs[:num_output_examples]],
    )

  individual_rules_issues = check_changes_claimed_for_rules(phonrule_params)

  individual_rules_msg = (
    "The following problems were noted with individual "
    "rules in your previous code."
  )

  if individual_rules_issues:
    if assessment:
      individual_rules_msg = (
        f"\nIn addition to the above, {individual_rules_msg.lower()}"
      )
    assessment.append(individual_rules_msg)
    assessment.append(individual_rules_issues)

  assessment = "\n".join(assessment).strip()

  critique = ""
  if assessment:
    critique += (
      "The issues below seem to point to bugs in your previous code in one " +
      "or more of your rules:\n\n" +
      assessment +
      "\n\n" +
      "Please correct the errors noted above."
    )
  if code_unchanged:
    msg = (
      "In addition to any other issues, it looks as if your code has "
      "not been changed at all from the previous round, despite what you "
      "may claim."
    )
    critique = f"{critique}\n\n{msg}"
  prompt = prompter.build_whole_prompt(
    basic_instructions=phonological_rules_basic_prompt,
    source_code=phonrule_params["source_code"],
    outputs_from_last_round=outputs_from_last_round,
    critique=critique,
    improvement_instructions=(
      prompter.PHONOLOGICAL_RULES_IMPROVEMENT_INSTRUCTIONS
    ),
    which_task="phonological_rules",
    which_attempt=which_attempt,
  )
  return prompt


def run_phonotactics_loop() -> None:
  """Performs an iterative run to refine phonotactics."""
  client = llm.client()
  initial_prompt = ""
  output_python_path = ""
  for it in range(MAX_ITER.value):
    if initial_prompt:
      user_prompt = build_phonotactics_critique_prompt(
        phonotactics=output_python_path,
        phonotactics_basic_prompt=initial_prompt,
        stress_placement=loader.STRESS_PLACEMENT.value,
        num_output_examples=NUM_OUTPUT_EXAMPLES.value,
        which_attempt=it + 1,
      )
    else:
      user_prompt = build_phonotactics_prompt(language=LANGUAGE.value)
      initial_prompt = user_prompt
    logging.info(user_prompt)
    base_name = f'{PHONOTACTICS_BASE.value}_{it:02d}'
    if USER_PROMPT_DUMP.value:
      with open(f'{base_name}_prompt.txt', "w") as stream:
        stream.write(f"{user_prompt}\n")
    full_output = llm.llm_predict(
      client,
      llm.MODEL.value,
      prompter.SYSTEM_PROMPT,
      user_prompt,
      max_tokens=4096,
    ).strip()
    output = full_output.split("<OUTPUT>")[-1].split("</OUTPUT>")[0].strip()
    # Added to overcome GPT stupidity when writing code:
    if output.startswith("```python") and output.endswith("```"):
      output = output.replace("```python", "")
      output = output[:-3].strip()
    finished = False
    if "NO FURTHER CHANGES" in output.upper():
      logging.info(f"No further changes to make, skipping writing new code.")
      finished = True
    else:
      output_python_path = f'{base_name}.py'
      logging.info(f"Writing output to {output_python_path}")
      with open(output_python_path, "w") as stream:
        stream.write(f"{output}\n")
    output_full = f'{base_name}.txt'
    logging.info(f"Writing full output to {output_full}")
    with open(output_full, "w") as stream:
      stream.write(f"{full_output}\n")
    if finished:
      break
  if output_python_path:
    msg = (
      "To test the phonotactics from the command line, run the following:\n\n" +
      f"python3 {output_python_path} --num_morphemes 10"
    )
    print(msg)


def run_phonological_rules_loop() -> None:
  """Performs an iterative run to refine rules."""
  client = llm.client()
  initial_prompt = ""
  output_python_path = ""
  code_unchanged = False
  for it in range(MAX_ITER.value):
    if initial_prompt:
      user_prompt = build_phonological_rules_critique_prompt(
        phonotactics=loader.PHONOTACTICS.value,
        phonrules=output_python_path,
        phonological_rules_basic_prompt=initial_prompt,
        stress_placement=loader.STRESS_PLACEMENT.value,
        num_output_examples=NUM_OUTPUT_EXAMPLES.value,
        which_attempt=it + 1,
        code_unchanged=code_unchanged,
      )
    else:
      user_prompt = build_phonological_rules_prompt(
        phonotactics=loader.PHONOTACTICS.value,
        nclosest=NUM_CLOSEST.value,
        stress_placement=loader.STRESS_PLACEMENT.value,
        num_examples=NUM_PHONOTACTICS_EXAMPLES.value,
      )
      initial_prompt = user_prompt
    logging.info(user_prompt)
    base_name = f'{PHONRULES_BASE.value}_{it:02d}'
    if USER_PROMPT_DUMP.value:
      with open(f'{base_name}_prompt.txt', "w") as stream:
        stream.write(f"{user_prompt}\n")
    full_output = llm.llm_predict(
      client,
      llm.MODEL.value,
      prompter.SYSTEM_PROMPT,
      user_prompt,
      max_tokens=4096,
    ).strip()
    output = full_output.split("<OUTPUT>")[-1].split("</OUTPUT>")[0].strip()
    # Added to overcome GPT stupidity when writing code:
    if output.startswith("```python") and output.endswith("```"):
      output = output.replace("```python", "")
      output = output[:-3].strip()
    finished = False
    if "NO FURTHER CHANGES" in output.upper():
      logging.info(f"No further changes to make, skipping writing new code.")
      finished = True
    else:
      if output_python_path:
        previous_code = open(output_python_path).read().strip()
        if output == previous_code:
          code_unchanged = True
        else:
          code_unchanged = False
      output_python_path = f'{base_name}.py'
      logging.info(f"Writing output to {output_python_path}")
      with open(output_python_path, "w") as stream:
        stream.write(f"{output}\n")
    output_full = f'{base_name}.txt'
    logging.info(f"Writing full output to {output_full}")
    with open(output_full, "w") as stream:
      stream.write(f"{full_output}\n")
    if finished:
      break
  if output_python_path:
    msg = (
      "To test the rules from the command line, run the following:\n\n" +
      "python3 agentic_phonology/test_rules.py " +
      f'--phonotactics "{loader.PHONOTACTICS.value}" ' +
      f'--phonrules "{output_python_path}" ' +
      f"--n 10"
    )
    print(msg)
