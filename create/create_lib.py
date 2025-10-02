"""Library tools for prompting the LLM to design some aspect of the language."""
import sys
import os

sys.path.append(os.path.abspath("."))

from absl import logging
from llm import llm
from typing import Dict
from utils import common_utils as cu


def run_model(
    client: llm.LLMClient,
    params: Dict[str, str],
    output_path: str,
    output_path_full: str,
    user_prompt_path: str,
    modular_morphosyntax: bool = False,
    user_prompt_dump: str = "",
) -> bool:
    """Perform one run of the model.

    Returns True if we should continue, False if we should stop.

    Args:
      client: A LLM client.
      params: A dictionary, parameters for the user prompt.
      output_path: Path to the output data.
      output_path_full: Path to the entire response.
      user_prompt_path: Path to user prompt.
      modular_morphosyntax: Whether or not the prompt constructor should
        use the modular_morphosyntax constructor.
      user_prompt_dump: Path to dump user prompt to.
    Returns:
      Boolean.
    """
    system_prompt = cu.create_system_prompt()
    user_prompt = cu.create_user_prompt(
        params,
        user_prompt_path=user_prompt_path,
        modular_morphosyntax=modular_morphosyntax,
    )
    logging.info(user_prompt)
    if user_prompt_dump:
        with open(user_prompt_dump, "w") as stream:
            stream.write(f"{user_prompt}\n")
    prediction = llm.llm_predict(
        client,
        llm.MODEL.value,
        system_prompt,
        user_prompt,
        max_tokens=4096,
    )
    prediction = prediction.strip()
    output = prediction.split("<OUTPUT>")[-1].split("</OUTPUT>")[0]
    output = output.strip()
    print("Output:", output)  # Debugging output
    # Added to overcome LLMs, such as GPT, that insist on sticking ```python at
    # the beginning of code despite being told not to:
    if output.startswith("```python") and output.endswith("```"):
        output = output.replace("```python", "")
        output = output[:-3].strip()
    proceed = True
    if output == "NO CHANGES TO MAKE.":
        proceed = False
        logging.info("No further changes to make, skipping writing new output.")
    else:
        logging.info(f"Writing output to {output_path}")
        with open(output_path, "w") as stream:
            stream.write(f"{output}\n")
    logging.info(f"Writing full prediction to {output_path_full}")
    with open(output_path_full, "w") as stream:
        stream.write(f"{prediction}\n")
    return proceed
