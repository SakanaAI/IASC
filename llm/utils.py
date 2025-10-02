"""Utilities."""
## THIS IS NOW DEPRECATED IN FAVOR OF ../utils/common_utils.py
import copy
import random
import re

COMMENT = re.compile("<!--.*?-->")


def load_system_instructions(
    system_instructions: str,
    verbose: bool=False,
) -> str:
  """Loads system instructions from file.

  Args:
    system_instructions: A path.
    verbose: boolean, whether to print out the instructions.
  Returns:
    A string containing the system instructions.
  """
  with open(system_instructions) as stream:
    instructions = stream.readlines()
    instructions = "".join(instructions)
    instructions = instructions.replace("<INSTRUCTIONS>", "")
    instructions = instructions.replace("</INSTRUCTIONS>", "")
    instructions = COMMENT.sub("", instructions)
    instructions = instructions.strip()
    return instructions
