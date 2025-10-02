"""Loading data from Python source files.

This extends the mechanism in ../utils/common_utils.py
"""

from absl import flags
from importlib.machinery import SourceFileLoader
from types import ModuleType


PHONOTACTICS = flags.DEFINE_string(
  "phonotactics",
  None,
  "Path to best phonotactics.",
)
PHONRULES = flags.DEFINE_string(
  "phonrules",
  None,
  "Path to phonological rules.",
)
STRESS_PLACEMENT = flags.DEFINE_string(
  "stress_placement",
  "penultimate",
  "Lexical stress placement",
)


def load_python_source_module(name: str, path: str) -> ModuleType:
  """Loads Python source from path as a module.

  **NB**: This is potentially dangerous. Also load_module() is deprecated so we
  need to investigate how to use exec_module() and create_module(), which have a
  different interface, because of course whenever there is a "X is deprecated,
  use Y" situation, the API for Y is completely different from that of X.

  Args:
    name: name for the module
    path: path to the predefined Python code.
  Returns:
    A module.

  """
  return SourceFileLoader(name, path).load_module()


def load_phonotactics(
    input_phonotactics: str,
    stress_placement: str="penultimate",
    idx: int=0,
):
  module__ = load_python_source_module(
    f"phonotactics_{idx}",
    input_phonotactics,
  )
  consonants = ", ".join([c for c in module__.consonants.keys()])
  vowels = ", ".join([c for c in module__.vowels.keys()])
  with open(input_phonotactics) as stream:
    source_code = stream.read().strip()
  return {
    "consonants": consonants,
    "vowels": vowels,
    "stress_placement": stress_placement,
    "morpheme_generator": module__.generate_morpheme,
    "source_code": source_code,
  }


def load_phonrules(input_phonrules: str, idx: int=0):
  module__ = load_python_source_module(f"phonrules_{idx}", input_phonrules)
  with open(input_phonrules) as stream:
    source_code = stream.read().strip()
  return {
    "rules": module__.rules,
    "module": module__,
    "source_code": source_code,
  }
