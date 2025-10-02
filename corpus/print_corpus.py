"""Pretty print corpora in JSONL format."""
import sys
import os
sys.path.append(os.path.abspath("."))

import corpus_management as lib
import jsonlines

from absl import app
from absl import flags

CORPORA = flags.DEFINE_list("corpora", "", "List of JSONL corpora.")


def main(unused_argv):
  for corpus in CORPORA.value:
    with jsonlines.open(corpus) as reader:
      text = lib.text_to_interlinear(reader)
      print("=" * 80)
      print(corpus)
      print(text)


if __name__ == "__main__":
  app.run(main)
