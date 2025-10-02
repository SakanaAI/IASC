import glob
import os
import sys
sys.path.append(os.path.abspath("."))

import agentic_phonology.loader as loader

from absl import app
from absl import flags
from absl import logging

N = flags.DEFINE_integer("n", 10, "Number of examples to generate.")
MARK_WORD_BOUNDARIES = flags.DEFINE_bool(
  "mark_word_boundaries",
  True,
  "Add word boundaries to beginning and end.",
)


def main(unused_argv):
  phonotactics_params = loader.load_phonotactics(
    loader.PHONOTACTICS.value,
    loader.STRESS_PLACEMENT.value,
  )
  files = sorted(glob.glob(loader.PHONRULES.value))
  phonrules_params = [
    (p, loader.load_phonrules(p, idx=i)) for i, p in enumerate(files)
  ]
  for _ in range(N.value):
    inp = phonotactics_params["morpheme_generator"]()
    if MARK_WORD_BOUNDARIES.value:
      inp = f"# {inp} #"
    for p, params in phonrules_params:
      try:
        out = params["rules"](inp)
      except Exception as e:
        logging.error(f"Exception in rules from {p}: {e}")
        out = "No output"
      tag = "  "
      if inp != out:
        tag = "* "
      print(f"{tag}/{inp}/ -> /{out}/")


if __name__ == "__main__":
  flags.mark_flag_as_required("phonotactics")
  flags.mark_flag_as_required("phonrules")
  app.run(main)
