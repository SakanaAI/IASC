import os
import sys
sys.path.append(os.path.abspath("."))

import agentic_phonology.run_phonology as rp

from absl import app
from absl import flags

WHICH_TASK = flags.DEFINE_enum(
  "which_task",
  "phonotactics",
  ["phonotactics", "phonrules"],
  "Which task to perform",
)

def main(unused_argv):
  if WHICH_TASK.value == "phonotactics":
    flags.mark_flag_as_required("language")
    flags.mark_flag_as_required("phonotactics_base")
    os.makedirs(os.path.dirname(rp.PHONOTACTICS_BASE.value), exist_ok=True)
    rp.run_phonotactics_loop()
  else:
    flags.mark_flag_as_required("phonotactics")
    flags.mark_flag_as_required("phonrules_base")
    os.makedirs(os.path.dirname(rp.PHONRULES_BASE.value), exist_ok=True)
    rp.run_phonological_rules_loop()


if __name__ == "__main__":
  app.run(main)
