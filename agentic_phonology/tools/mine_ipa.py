"""Extract IPA transcriptions from Wiktionary.
"""
import collections
import csv
import jsonlines
import os

from absl import app
from absl import flags
from absl import logging

HOME = os.environ["HOME"]

WIKI = flags.DEFINE_string(
  "wiki",
  None,
  "Path to Wiktionary dump in the form of raw-wiktextract-data.jsonl "
  "from https://kaikki.org/dictionary/rawdata.html.",
)
LANGUAGE = flags.DEFINE_string("language", None, "Language.")
OUTDIR = flags.DEFINE_string("outdir", None, "Output directory for CSVs.")
MINCNT = flags.DEFINE_integer(
  "mincnt",
  500,
  "Minimum number of transcriptions required.",
)


KEYS = ["lang", "word", "sounds"]


def main(unused_argv):

  os.makedirs(OUTDIR.value, exist_ok=True)

  def collect(elt):
    for k in KEYS:
      if k not in elt:
        return False
    if elt["lang"] == LANGUAGE.value or LANGUAGE.value == "any":
      return True
    return False

  pronunciations = collections.defaultdict(set)
  with jsonlines.open(WIKI.value) as reader:
    for elt in reader:
      if collect(elt):
        word = elt["word"]
        lang = elt["lang"]
        for trans in elt["sounds"]:
          if "ipa" in trans:
            ipa = trans["ipa"]
            pronunciations[lang].add(ipa)
  for lang in pronunciations:
    if len(pronunciations[lang]) < MINCNT.value:
      logging.info(f"Skipping {lang} since it has too few examples.")
      continue
    rows = [[lang, ipa] for ipa in pronunciations[lang]]
    base = lang.replace(" ", "_")
    with open(f"{OUTDIR.value}/{base}.csv", "w") as stream:
      writer = csv.writer(stream, delimiter=",", quotechar='"')
      writer.writerows(rows)


if __name__ == "__main__":
  flags.mark_flag_as_required("language")
  flags.mark_flag_as_required("outdir")
  flags.mark_flag_as_required("wiki")
  app.run(main)
