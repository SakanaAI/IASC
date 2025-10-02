"""Corpus management tools."""
import sys
import os
sys.path.append(os.path.abspath("."))

import jsonlines
import os
import re
import string

import corpus.lexicon as lexicon

from absl import app
from absl import flags
from typing import Sequence

Text = lexicon.Text
AFFIX = lexicon.AFFIX
PUNCT = lexicon.PUNCT
PREPUNCT = re.compile(r"([\.,\?\"!:;])([A-Za-z])")  # right-attached punctuation


def _readlines(path):
  with open(path) as s:
    text = s.read().strip()
    return [l.strip() for l in text.split("\n")]


def interleave(
    source_text_path: str,
    gloss_path: str,
) -> Text:
  """Interleave original source text and its translation.

  Args:
    source_text_path: Path to source text
    gloss_path: Path to glosses.
    outdir: output_directory.
  Returns:
    Sequence of dictionaries with source text and parsed morphosyntactic glosses.
  """
  source_text = _readlines(source_text_path)
  gloss = _readlines(gloss_path)
  L = len(source_text)
  assert len(source_text) == len(gloss)
  result = []

  # TODO(rws): This is a mess with lots of special-casing. Come up with a more
  # elegant way to do this.
  def word_parse(w):
    w = w.strip("-")
    if len(w) == 1:
      return [w]
    L = len(w)
    new_w = [w[0]]
    prefixing = True
    legal_marker_chars = string.ascii_uppercase + string.digits
    for i in range(1, len(w) - 1):
      if w[i] == "-":
        if (
            w[i - 1] not in legal_marker_chars and
            w[i + 1] not in legal_marker_chars
        ):
          new_w.append("-")
        elif prefixing and w[i - 1] in legal_marker_chars:
          new_w.append("-")
          new_w.append(" ")
        else:
          new_w.append(" ")
          new_w.append("-")
      else:
        new_w.append(w[i])
      if w[i] in string.ascii_lowercase:
        prefixing = False
    new_w.append(w[-1])
    final_w = []
    for s in "".join(new_w).split():
      if s == "-":
        continue
      if s.startswith("-") and s[1] not in legal_marker_chars:
        final_w.append(s[1:])
      elif s.endswith("-") and s[-2] not in legal_marker_chars:
        final_w.append(s[:-1])
      else:
        final_w.append(s)
    return final_w

  def parse(morphosyntax):
    morphosyntax = PREPUNCT.sub(r"\1_ \2", morphosyntax)
    morphosyntax = PUNCT.sub(r" \1", morphosyntax)
    return [word_parse(w) for w in morphosyntax.split()]

  for i in range(L):
    result.append({
      "source_text_path": source_text_path,
      "line": i,
      "source": source_text[i],
      "morphosyntax": parse(gloss[i]),
    })
  return result


def write_corpus(corpus: Sequence[Text], outdir: str):
  for text in corpus:
    for elt in text:
      if "source_text_path" in elt:
        source_text_path = elt["source_text_path"]
        break
    outfile = os.path.basename(source_text_path).replace(".txt", ".jsonl")
    outfile = os.path.join(outdir, outfile)
    with jsonlines.open(outfile, "w") as s:
      s.write_all(text)


def text_to_interlinear(text: Text, include_source: bool=True) -> str:
  lines = []
  for elt in text:
    segment = []
    if include_source:
      source = elt["source"]
      segment.append(source)
    gloss = " ".join(["".join(m) for m in elt["morphosyntax"]])
    if not gloss:
      continue
    segment.append(gloss)
    # TODO(rws): Generalize this when we add rules.
    deep_phonology = " ".join(
      [p.replace(" ", "") for p in elt["deep_phonology"]],
    )
    segment.append(deep_phonology)
    if "printed_form" in elt:
      printed_form = elt["printed_form"]
    else:
      printed_form = ""
    segment.append(printed_form)
    segment = "\n".join(segment)
    lines.append(segment)
  return "\n\n".join(lines)
