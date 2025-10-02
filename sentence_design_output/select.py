"""Select subset to fit into about 2,000 characters"""

import glob
import random
import re

NUM = re.compile(r"^\d+\. +")

OUTPUT = "sentence_design_output/grammatical_test_sentences.txt"

with open(OUTPUT, "w") as out:
  for path in sorted(glob.glob("sentence_design_output/*.txt")):
    if path.endswith("_full.txt"):
      continue
    with open(path) as inp:
      text = inp.read()
      random.seed(len(text))
      lines = [l.strip() for l in text.split("\n") if l]
      random.shuffle(lines)
      lines = [NUM.sub("", l) for l in lines[:5]]
      for line in lines:
        line = line.replace("(INCLUSIVE)", "").replace("(EXCLUSIVE)", "").strip()
        out.write(f"{line}\n\n")
