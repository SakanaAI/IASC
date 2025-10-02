"""Prompt builder for iterative agentic phonology."""
from jinja2 import Template

# Copy of prompts/system_prompt.txt

SYSTEM_PROMPT = """

You are an expert linguist who is also an expert Python programmer.  You know a
wide variety of languages, and your hobby is creating Constructed Languages
--- ConLangs.

""".strip()


def template(templ: str):
  return Template(templ.strip())


PROMPT_META_TEMPLATE = template(
"""

{{ basic_instructions }}

{% if outputs_from_last_round != "" %}

Here is the code from your last round:

<CODE>
{{ source_code }}
</CODE>

{% if which_task == "phonotactics" %}

Here are some outputs from your last phonotactic model:

{% else %}

Here are some inputs and outputs from your last round of rules:

{% endif %}

{{ outputs_from_last_round }}

{{ critique }}

{{ improvement_instructions }}

Note that this is round {{ which_attempt }}. If this is more than 5 rounds and
if the above assessment does not find any bugs that you need to fix, then PLEASE
STOP and output NO FURTHER CHANGES as instructed above.

{% endif %}

"""
)

# Copied more or less verbatim from ../prompts/phonotactics.txt but with the
# addition to require the insertion of syllable boundaries. Word boundaries on
# the other hand will be inserted in code when the phonotactics output gets
# passed to the phonological rule phase.
PHONOTACTICS_BASIC_INSTRUCTIONS = template(
"""
The first phase in creating any ConLang is to create a set of words for the
language. Typically this involves defining the phoneme set, including some
information about their relative frequency as well as their phoneme classes
(consonant, vowel, tone) and phonetic features.

Then one needs to define a set of morpheme templates, which define possible
morpheme shapes. This could be as simple as "CVC" for a language that allows CVC
syllables, to something more complex like "CCVC.CCVC" for languages that allow
disyllabic morphemes with each syllable of the form "CCVC".

We are going to create a language that has a phoneme set and morpheme shapes
that are similar to those of {{ language }}. In this first phase, I want you to
design a set of phonemes similar to those of {{ language }}, with similar
relative frequencies for the phonemes to those of that language. All phonemes
should be represented in the International Phonetic Alphabet (IPA).

Then you will design a set of morpheme templates similar to the patterns found
in {{ language }}.

Your design should be implemented as a standalone Python program that takes as a
command-line argument the number of morpheme forms to generate. For example:

python3 your_program.py --num_morphemes 100

would randomly generate 100 morphemes.

Your code should have a function `generate_morpheme` that takes no arguments,
and returns a single morpheme.

There are two main points to pay attention to.

1) Within each morpheme, please place spaces between every phoneme

2) Mark internal syllable boundaries with ".", again separated by spaces.

For example, the following is a valid output.

"o . k a . tʃ a"

The following would be wrong, because the phonemes are not space-separated:

"o.ka.tʃa"

Your code should include two global dictionaries, one named "consonants" and the
other named "vowels". These should contain the phonemes mapped to their relative
frequencies. Thus for example, a partial consonant dictionary might include:

consonants = {"m": 3, "n": 7 ...}

Please make sure you include needed functions and imports in your code.

Finally, place your resulting code in the block <OUTPUT></OUTPUT>.
PLEASE DO NOT PUT ANYTHING EXCEPT THE PYTHON CODE WITHIN THIS BLOCK.

This task will depend on your deep knowledge of linguistics, and Python
programming.  Good luck!
"""
)

PHONOTACTICS_IMPROVEMENT_INSTRUCTIONS = """

Apart from correcting any errors noted above, please also consider the outputs
from the last round to see if your phonotactic rules could be improved.

In particular please check this output and see if there are any words that look
phonotactically odd. Are there words or syllables that begin with
rather unlikely consonant sequences? For example, cross-linguistically, the
following syllables have strange onsets since they begin with consonant
sequences, or end with consonant sequences that are unlikely onsets or codas
given the Sonority Hierarchy:

p d ɛ d p
m d ɔʏ t
t n ɔ t

Also check your syllable boundary assignments.

There may also be phonemes that are particularly characteristic of the language
that are missing from your previous code, or ones that should not be
there. Please also check for that.

As before, place your resulting code in the block <OUTPUT></OUTPUT>.

ONLY if no errors are noted, and if your assessment of the rules is that they
are adequate, then instead of code return the following:

<OUTPUT>NO FURTHER CHANGES</OUTPUT>

""".strip()


## A silly hack motivated by Alper et al. 2025,
## https://arxiv.org/html/2508.06094v1
PHOTOTACTICS_BASIC_INSTRUCTIONS = template(
"""
The first phase in creating any ConLang is to create a set of words for the
language. Typically this involves defining the phoneme set, including some
information about their relative frequency as well as their phoneme classes
(consonant, vowel, tone) and phonetic features.

Then one needs to define a set of morpheme templates, which define possible
morpheme shapes. For a human language based on sounds, this could be as simple
as "CVC" for a language that allows CVC syllables, to something more complex
like "CCVC.CCVC" for languages that allow disyllabic morphemes with each
syllable of the form "CCVC".

We are going to create a language for an alien species that uses color rather
than sound to communicate, flashing one color after another. The photeme set of
this language consists of the following elements, single letters representing
numbers as follows:

R   red
O   orange
Y   yellow
G   green
B   blue
I   indigo
V   violet

Your design should be implemented as a standalone Python program that takes as a
command-line argument the number of morpheme forms to generate. For example:

python3 your_program.py --num_morphemes 100

would randomly generate 100 morphemes.

Your code should have a function `generate_morpheme` that takes no arguments,
and returns a single morpheme.

There are two main points to pay attention to.

1) Within each morpheme, please place spaces between every photeme

2) Mark internal syllable boundaries with ".", again separated by spaces.

For example, the following is a valid output.

"R . R G . B I"

The following would be wrong, because the phonemes are not space-separated:

"R.RG.BI"

You will need to decide which of the photemes represents "consonants" and which
"vowels". Your code should include two global dictionaries, one named
"consonants" and the other named "vowels". These should contain the phonemes
mapped to their relative frequencies. Thus for example, a partial consonant
dictionary might include:

consonants = {"R": 3, "R": 7 ...}

Please make sure you include needed functions and imports in your code.

Finally, place your resulting code in the block <OUTPUT></OUTPUT>.
PLEASE DO NOT PUT ANYTHING EXCEPT THE PYTHON CODE WITHIN THIS BLOCK.

This task will depend on your deep knowledge of linguistics, and Python
programming.  Good luck!
"""
)

PHOTOTACTICS_IMPROVEMENT_INSTRUCTIONS = """

Apart from correcting any errors noted above, please also consider the outputs
from the last round to see if your phototactic rules could be improved.

In particular please check this output and see if there are any words that look
"phototactically" odd.

Also check your syllable boundary assignments.

There may also be photemes that are particularly characteristic of the language
that are missing from your previous code, or ones that should not be
there. Please also check for that.

As before, place your resulting code in the block <OUTPUT></OUTPUT>.

ONLY if no errors are noted, and if your assessment of the rules is that they
are adequate, then instead of code return the following:

<OUTPUT>NO FURTHER CHANGES</OUTPUT>

""".strip()


PHONOLOGICAL_RULES_BASIC_INSTRUCTIONS = template(
"""
Languages change and one of the ways in which they change is for their sound
systems to change.

Recall that in our language the set of phonemes (in IPA), not including the
stress, syllable-, morpheme-, and word-boundary markers, is as follows:

Consonants: {{ consonants }}
Vowels: {{ vowels }}

Here are some sample words generated from your phonotactic grammar:

{{ sample_words }}

We compared this set against a range of reconstructed languages, and found
{% if num == 1 %} one language which seems closest in terms of phonological makeup.
We give the name of the reconstructed language, its reconstructed
phoneset and one or more sets of rules that have been reconstructed to derive
daughter languages from this language.
{% else %} {{ num }} languages which seemss closest in terms of phonological makeup. For each
of these, we give the name of the reconstructed language, its reconstructed
phoneset and one or more sets of rules that have been reconstructed to derive
daughter languages from those languages.
{% endif %}

{{ reconstructions }}

Based on the above:

1. Devise a set of rules that makes sense for the phoneme set of our language
   given the observed phonotactics.

2. Implement each of these rules in Python.

3. Put the rules together into a Python library.

Each rule should take the form of a python function that takes as input a string
of space-delimited phonemes and possible syllable boundaries, and outputs a
string in the same format.  You must make sure that the output phonemes are
space-delimited as in the input. All your rules must allow for the presence of
spaces in the input!!!!!

We assume "#" as an end-of-word symbol: you should assume that word boundaries
in the input will be indicated by this symbol.  The symbol ".", if it occurs,
represents a syllable boundary. The symbol "-" if it occurs represents a
morpheme boundary. For rules that depend on stress, you are to
assume that the language has {{ stress_placement }} stress.

Start each rule with a comment that gives some examples of what the input and
output of the rule should be. See below for format.

The template for a rule will be:

## BEGIN TEMPLATE
def ruleName1(inp: str) -> str:
  # Example inputs/outputs:
  # input1 -> output1
  # input2 -> output2
  ...
  return output
## END TEMPLATE

Note that any phonological rules that should apply across syllables MUST take
the possible presence of "." into account.  Thus a nasal assimilation rule
within a word should normally apply across syllables, so that:

a n . b o n

would become

a m . b o n

This means that you will need to include the possibility of an intervening space
plus "." in your regular expressions.

Similarly, morpheme boundaries ("-") should NOT generally affect the output of
rules and so rules must be written to apply across them.

Implement a wrapper function that calls your set of rules in the intended order
of their application:

## BEGIN TEMPLATE
def rules(inp: str) -> str:
  output = ruleName1(inp)
  output = ruleName2(output)
  # ...
  return output
## END TEMPLATE

Remember:

1. All phonemes are space-separated. All your rules must take this into
   account. Be careful: make sure you allow for EXACTLY ONE space between
   phonemes since rules that depend on more than one space being there will not
   work. Your rules must also output space-separated phonemes. Thus if the
   input to a nasal assimilation rule is "n p", the output should be "m p", not
   "mp".

2. If you use the Python regex library and make use of groups, make sure you
   have enough capturing groups to support the number of back references you
   assume. A common error is to have backreferences like "\\1\\2", but only have a
   single previous capturing group.

3. Avoid using regex look-behind since you inevitably miss the point that
   look-behind patterns are fixed width, which triggers the
   "sre_constants.error: look-behind requires fixed-width pattern" error.

4. Finally, it is OK if your rules introduce phonemes that are NOT in the input
   phoneme set since, after all, that is what sound change is all about.

Make sure you have imports for all needed libraries in your code.

Explain your reasoning. Then place your resulting code in the block
<OUTPUT></OUTPUT>.

This task will depend on your deep knowledge of historical linguistics, and
Python programming.  Be creative and good luck!

""")


PHONOLOGICAL_RULES_IMPROVEMENT_INSTRUCTIONS = """

Apart from correcting any errors noted above, please also consider the outputs
from the last round to see if your rules could be improved.

Pay particular attention to whether the rules seem to be applying correctly: are
there any rules that should apply in a particular case given what they are
supposed to do, but apparently are not applying?

As before, place your resulting code in the block <OUTPUT></OUTPUT>.

If no errors are noted, and if your assessment of the rules is that they
are adequate, then instead of code return the following:

<OUTPUT>NO FURTHER CHANGES</OUTPUT>

""".strip()


def clean_whitespace(prompt):
  lines = prompt.split("\n")
  new_prompt = []
  prev = ""
  # NB: Don't actually strip most lines since we want to preserve code indents.
  for line in lines:
    empty_line = line.strip()
    if not empty_line and not prev:
      continue
    else:
      if not empty_line:
        new_prompt.append(empty_line)
      else:
        new_prompt.append(line)
    prev = empty_line
  return "\n".join(new_prompt).strip()


def build_whole_prompt(
    basic_instructions: str,
    source_code: str="",
    outputs_from_last_round: str="",
    critique: str="",
    improvement_instructions: str="",
    which_task: str="phonological_rules",
    which_attempt: int=1,
):
  return clean_whitespace(
    PROMPT_META_TEMPLATE.render(
      basic_instructions=basic_instructions,
      source_code=source_code,
      outputs_from_last_round=outputs_from_last_round,
      critique=critique,
      improvement_instructions=improvement_instructions,
      which_task=which_task,
      which_attempt=which_attempt,
    )
  )
