"""Creates language-specific ngram LMs.
"""

import collections
import csv
import glob
import os

from absl import app
from absl import flags
from absl import logging
from jinja2 import Template

IPA_DIR = flags.DEFINE_string(
  "ipa_dir",
  "agentic_phonology/data/ipas",
  "Path to IPA data extracted with mine_ipa.py",
)
SYMBOLS = flags.DEFINE_string(
  "symbols",
  "agentic_phonology/data/ipas/lm.syms",
  "Path to LM symbol table",
)
ORDER = flags.DEFINE_integer("order", 3, "Ngram order")


def load_ipas():
  paths = glob.glob(f"{IPA_DIR.value}/*.csv")
  ipas = collections.defaultdict(set)

  def clean_ipa(ipa):
    ipa = ipa.replace("/", "")
    ipa = ipa.replace("[", "")
    ipa = ipa.replace("]", "")
    ipa = "".join(ipa.split())
    return ipa

  for path in paths:
    with open(path) as stream:
      reader = csv.reader(stream, delimiter=",", quotechar='"')
      for row in reader:
        lang, ipa = row
        if "(" in ipa or ")" in ipa:
          continue
        ipa = clean_ipa(ipa)
        lang = lang.replace(" ", "_")
        ipas[lang].add(ipa)
  return ipas


FARCOMPILESTRINGS = Template(
"""
farcompilestrings --fst_type=compact \
                  --symbols={{symbols}} \
                  --keep_symbols "{{text}}" "{{far}}"
""".strip()
)
NGRAMCOUNT = Template(
"""

ngramcount --order={{order}} "{{far}}" "{{cnts}}"

""".strip()
)
NGRAMMAKE = Template(

"""

ngrammake "{{cnts}}" "{{mod}}"


""".strip()

)


def make_ngram_models(ipas):
  symbols = set()
  for lang in sorted(ipas):
    path_base = f"{IPA_DIR.value}/{lang}"
    text = f"{path_base}.txt"
    with open(text, "w") as stream:
      for ipa in ipas[lang]:
        transcription = " ".join([c for c in ipa] + ["</s>"])
        stream.write(f"{transcription}\n")
    far = f"{path_base}.far"
    cmd = FARCOMPILESTRINGS.render(symbols=SYMBOLS.value, text=text, far=far)
    os.system(cmd)
    cnts = f"{path_base}.cnts"
    cmd = NGRAMCOUNT.render(order=ORDER.value, far=far, cnts=cnts)
    os.system(cmd)
    mod =  f"{path_base}.mod"
    cmd = NGRAMMAKE.render(cnts=cnts, mod=mod)
    os.system(cmd)
    logging.info(f"Created {mod}")


def make_symbol_table(ipas):
  symbols = set()
  for lang in ipas:
    for ipa in ipas[lang]:
      for c in ipa:
        symbols.add(c)
  with open(SYMBOLS.value, "w") as stream:
    stream.write("<epsilon>\t0\n")
    stream.write("</s>\t1\n")
    for i, c in enumerate(symbols):
      stream.write(f"{c}\t{i + 2}\n")
    stream.write(f"<UNK>\t{len(symbols) + 2}\n")


def main(unused_argv):
  ipas = load_ipas()
  make_symbol_table(ipas)
  make_ngram_models(ipas)


if __name__ == "__main__":
  app.run(main)
