import os
import sys
sys.path.append(os.path.abspath("../.."))

import collections
import jsonlines

from absl import app
from absl import flags
from absl import logging
from llm import llm
from jinja2 import Template


ABBREVIATIONS = flags.DEFINE_string(
  "abbreviations",
  "abbreviations.txt",
  "Path to text file with abbreviations and conventions.",
)
DIACHRONICA = flags.DEFINE_string(
  "diachronica",
  "diachronica.jsonl",
  "Path to JSONL with Diachronica data.",
)
PREVIOUS_OUTPUT = flags.DEFINE_string(
  "previous_output",
  None,
  "Path to previous output JSONL.",
)
OUTPUT = flags.DEFINE_string(
  "output",
  "output.jsonl",
  "Path to output JSONL.",
)
PRINT_FIRST_PROMPT = flags.DEFINE_bool(
  "print_first_prompt",
  False,
  "Print first prompt",
)


def load_abbreviations():
  with open(ABBREVIATIONS.value) as s:
    return s.read().strip()


def load_parent_rule_sets():
  with jsonlines.open(DIACHRONICA.value) as reader:
    elts = [e for e in reader]
  parent_rulesets = collections.defaultdict(list)
  for elt in elts:
    parent = elt["link"].split(" to ")[0].strip()
    parent_rulesets[parent].append((elt["link"], elt["rules"]))
  return parent_rulesets


def load_chains_and_rules():
  productions = {}
  phoneme_sets = {}
  with jsonlines.open(DIACHRONICA.value) as reader:
    elts = [e for e in reader]

  for elt in elts:
    if elt["proto_group"] and elt["proto_phonemes"]:
      phoneme_sets[elt["proto_group"]] = elt["proto_phonemes"]
    if elt["rules"]:
      productions[elt["link"]] = elt["rules"]

  parents = set()
  for elt in elts:
    for link in elt["chain"]:
      parent = link.split(" to ")[0].strip()
      parents.add(parent)

  leaves = set()
  for elt in elts:
    for link in elt["chain"]:
      daughter = link.split(" to ")[1].strip()
      if daughter not in parents:
        leaves.add(daughter)

  completed_links = set()
  for elt in elts:
    for link in elt["chain"]:
      parent, daughter = [e.strip() for e in link.split(" to ")]
      # If a daughter is a leaf in the tree it means there are no derived
      # languages in the set, which means we don't need to infer what the
      # daughter's phoneme set looks like, since no rules will get applied to
      # it.
      if daughter in leaves:
        continue
      if parent in phoneme_sets:
        if link not in completed_links:
          phonemes = ", ".join(phoneme_sets[parent])
          rules = "\n".join(productions[link])
          yield (
            {
              "parent": parent,
              "daughter": daughter,
              "phonemes": phonemes,
              "rules": rules,
            },
            phoneme_sets,
          )
        completed_links.add(link)
      else:
        pass


def load_previous_outputs():
  table = collections.defaultdict(list)
  if PREVIOUS_OUTPUT.value is not None:
    with jsonlines.open(PREVIOUS_OUTPUT.value) as reader:
      for elt in reader:
        table[elt["parent"], elt["daughter"]] = elt["daughter_phonemes"]
  return table


SYSTEM_PROMPT = "You are an expert historical linguist."

INSTRUCTIONS = Template(

"""

In what follows you will find a set of phonemes for a reconstructed language, in
this case {{ parent }}. You will then be given a set of phonological rules that
have been reconstructed to derive the daughter language {{ daughter }}.

To the best of your ability, given the phoneme set and the rules, derive a
probable phoneme set for the daughter language.  If you happen to know the
accepted reconstruction of the phoneme set for {{ daughter }}, you may let that
guide your reconstruction.

The reconstruction will necessarily be approximate since the rules are not
necessarily complete and are often vaguely specified, but here are some
guidelines:

If a rule is listed without a context, e.g.

e → i

then you may assume that all instances of /e/ are replaced with /i/.  If on the
other hand it is listed with a context, e.g.

e → i / _ #

then you should assume that only some instances of /e/ are changed and /e/ is
still retained in the system. Note that in some cases conditions on rules are
described in parenthetical remarks after the main body of the rule.

You will also occasionally see multistage rules like

dʒ → tʃ → ʃ

This can be interpreted as equivalent to

dʒ → ʃ

Sometimes a rule will specify changes in parallel, e.g.:

bʱ dʱ ɡʱ → β ð ɣ

which is to be interpreted the same as

bʱ → β
dʱ → ð
ɡʱ → ɣ

Unfortunately these rule sets are rather sloppily done so you will have to use
your beset judgment.

The following abbreviations and other conventions may be useful in understanding
some of the rules:

## BEGIN ABBREVIATIONS
{{ abbreviations }}
## END ABBREVIATIONS

The reconstructed phonemes for {{ parent }} are as follows:

{{ phonemes }}

The reconstructed rules deriving {{ daughter }} from {{ parent }} are as
follows:

## BEGIN RULES
{{ rules }}
## END RULES

Place your predicted phoneme set as a comma-separated list in a
<PHONEMES></PHONEMES> tag.

""".strip()

)

INSTRUCTIONS_ROUND_2 = Template(

"""

In what follows you will find a set of phonemes for a reconstructed language, in
this case {{ parent }}. You will then be given a set of phonological rules that
have been reconstructed to derive the daughter language {{ daughter }}.

To the best of your ability, given the phoneme set and the rules, derive a
probable phoneme set for the daughter language.  If you happen to know the
accepted reconstruction of the phoneme set for {{ daughter }}, you may let that
guide your reconstruction.

The reconstruction will necessarily be approximate since the rules are not
necessarily complete and are often vaguely specified, but here are some
guidelines:

If a rule is listed without a context, e.g.

e → i

then you may assume that all instances of /e/ are replaced with /i/.  If on the
other hand it is listed with a context, e.g.

e → i / _ #

then you should assume that only some instances of /e/ are changed and /e/ is
still retained in the system. Note that in some cases conditions on rules are
described in parenthetical remarks after the main body of the rule.

You will also occasionally see multistage rules like

dʒ → tʃ → ʃ

This can be interpreted as equivalent to

dʒ → ʃ

Sometimes a rule will specify changes in parallel, e.g.:

bʱ dʱ ɡʱ → β ð ɣ

which is to be interpreted the same as

bʱ → β
dʱ → ð
ɡʱ → ɣ

Unfortunately these rule sets are rather sloppily done so you will have to use
your beset judgment.

The following abbreviations and other conventions may be useful in understanding
some of the rules:

## BEGIN ABBREVIATIONS
{{ abbreviations }}
## END ABBREVIATIONS

The reconstructed phonemes for {{ parent }} are as follows:

{{ phonemes }}

The reconstructed rules deriving {{ daughter }} from {{ parent }} are as
follows:

## BEGIN RULES
{{ rules }}
## END RULES

On a previous round you constructed the following phoneme set for the
daughter language {{ daughter }}:

{{ daughter_phonemes }}

Have another look at the parent phoneme set for {{ parent }} above, the given
rules, and check if anything is missing from your proposed set for
{{ daughter }}, or if there are phonemes that should not be there. In
particular, be on the look-out for cases where a context-dependent rule might
have been overapplied to eliminate a parent-language phoneme that should NOT
have changed in all cases. Explain your reasons for any changes.

As before, place your new predicted phoneme set as a comma-separated list in a
<PHONEMES></PHONEMES> tag.

""".strip()

)


def construct_phoneme_sets(
    client: llm.LLMClient,
    print_prompt: bool=False,
):
  abbreviations = load_abbreviations()
  rule_sets = load_parent_rule_sets()
  previous_outputs = load_previous_outputs()
  for spec, phoneme_sets in load_chains_and_rules():
    parent = spec["parent"]
    daughter = spec["daughter"]
    phonemes = spec["phonemes"]
    rules = spec["rules"]
    previous_output = previous_outputs[parent, daughter]
    if previous_output:
      daughter_phonemes = ", ".join(previous_output)
      user_prompt = INSTRUCTIONS_ROUND_2.render(
        parent=parent,
        daughter=daughter,
        phonemes=phonemes,
        rules=rules,
        abbreviations=abbreviations,
        daughter_phonemes=daughter_phonemes,
      )
    else:
      user_prompt = INSTRUCTIONS.render(
        parent=parent,
        daughter=daughter,
        phonemes=phonemes,
        rules=rules,
        abbreviations=abbreviations,
      )
    if print_prompt:
      print(user_prompt)
      print_prompt = False
    output = llm.llm_predict(
      client,
      llm.MODEL.value,
      SYSTEM_PROMPT,
      user_prompt,
      max_tokens=4096,
    )
    daughter_phonemes = output.split("<PHONEMES>")[-1].split("</PHONEMES>")[0]
    daughter_phonemes = [p.strip() for p in daughter_phonemes.split(",")]
    daughter_phonemes = list(set(daughter_phonemes))
    phoneme_sets[daughter] = daughter_phonemes
    with jsonlines.open(OUTPUT.value, "a") as writer:
      elt = {
        "parent": parent,
        "daughter": daughter,
        "parent_phonemes": [p.strip() for p in phonemes.split(",")],
        "parent_to_daughter_rules": [r.strip() for r in rules.split("\n")],
        "daughter_phonemes": daughter_phonemes,
        "daughter_rule_sets": rule_sets[daughter],
        "full_output": output,
      }
      if previous_output:
        elt["previous_daughter_phonemes"] = previous_output
      logging.info(f"Writing phonemes for {daughter} (<{parent})")
      writer.write(elt)


def main(unused_argv):
  client = llm.client()
  construct_phoneme_sets(
    client,
    print_prompt=PRINT_FIRST_PROMPT.value,
  )


if __name__ == "__main__":
  app.run(main)
