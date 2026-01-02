"""Interface to various LLMs."""

import botocore
import boto3
import os
import torch

from absl import flags
from absl import logging

from botocore.config import Config # for Anthropic models
from botocore.exceptions import ClientError
from openai import OpenAI # for GPT models
from google import genai # for Gemini models
from time import sleep
from typing import Union, Optional
from vllm import LLM, SamplingParams

CLAUDE_WAIT_TIME = flags.DEFINE_integer(
    "claude_wait_time",
    5,
    "Number of seconds to wait if Claude is busy",
)
MODEL = flags.DEFINE_enum(
    "model",
    "claude",
    [
        "gpt-4o-mini",
        "gpt-4o-2024-08-06",
        "gpt-4-1106-preview",
        "gpt-4.1", # $2/1M input, $8/1M output. Snapshot gpt-4.1-2025-04-14
        "gpt5nano", #$ 0.05/1M input, $0.40/1M output
        "gpt-5-nano", # alias
        "gpt-5-mini", # $0.25/1M input, $2/1M output
        "gpt-5", # 1.250/1M input, $10.00/1M output
        "claude", # defaults to claude-sonnet-3-5
        "claude-sonnet-3-5",
        "claude-3-5-sonnet", # alias
        "claude-sonnet-4-5", # $3/1M input, $15/1M output
        "claude-4-5-sonnet", # alias
        "qwen",
        "llama",
        "gemini-2.5-flash", # 0.30/1M input, $2.50/1M output
        "gemini-2.5-pro"
    ],
    "Model to use for eval",
)
OPEN_AI_API_KEY = flags.DEFINE_string(
    "open_ai_api_key",
    None,
    "Your OpenAI API key",
)
TEMPERATURE = flags.DEFINE_float("temperature", 0.0, "LLM temperature")

QWEN_PATH = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"

LLAMA_PATH = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

LLMClient = Union[LLM, OpenAI, botocore.client.BaseClient]

model_names = {
    "gpt5nano": "gpt-5-nano-2025-08-07",
}


def llm_predict(
    client: LLMClient,
    model_name: str,
    system_instructions: str,
    user_prompt: str,
    max_tokens: int = 2048,
    reasoning_effort: Optional[str] = "medium",
) -> str:
    """Run the LLM to predict given the instructions and user prompt.

    Args:
      client: LLM client.
      model_name: Name of the model.
      system_instructions: System-level instructions.
      user_prompt: A user prompt.
      max_tokens: Maximum number of tokens for the LLM.
    Returns:
      The LLM response as a string.
    """
    if "claude" in model_name:
    # if model_name == "claude":
        if model_name in {"claude", "claude-3-5-sonnet", "claude-sonnet-3-5"}:
            model_id = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
        elif model_name in {"claude-4-5-sonnet", "claude-sonnet-4-5"}:
            model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        # Does this do anything?
        resource_arn = f"arn:aws:bedrock:us-east-1::foundation-model/{model_id}"
        # Claude does not like conversations that start with a system/assistant
        # message so we will punt on that, glom the two together, and hope for the
        # best. :/
        instructions = system_instructions + "\n" + user_prompt
        messages = [
            {
                "role": "user",
                "content": [{"text": instructions}],
            },
        ]

        def bad_response(exception):
            if "ServiceUnavailableException" in str(exception):
                return True
            if "Read timeout on endpoint URL" in str(exception):
                return True
            if "An error occurred" in str(exception):
                return True
            return False

        def wrap_claude_call(messages, model_id, max_tokens):
            while True:
                try:
                    response = client.converse(
                        modelId=model_id,
                        messages=messages,
                        inferenceConfig={
                            "maxTokens": max_tokens,
                            "temperature": TEMPERATURE.value,
                            "topP": 1.0,
                        },
                    )
                    return response["output"]["message"]["content"][0]["text"]
                except (ClientError, Exception) as e:
                    logging.info(
                        f"ERROR: Can't invoke '{model_id}'. Reason: {e}",
                    )
                    if bad_response(e):
                        logging.info(
                            f"Retrying in {CLAUDE_WAIT_TIME.value}",
                        )
                        sleep(CLAUDE_WAIT_TIME.value)
                    else:
                        return "**ERROR**"

        return wrap_claude_call(messages, model_id, max_tokens)
    elif model_name in ["qwen", "llama"]:
        tokenizer = client.get_tokenizer()
        sampling_params = SamplingParams(
            temperature=TEMPERATURE.value,
            top_p=0.95,
            max_tokens=max_tokens,
            n=1,
            seed=1,
        )
        messages = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": system_instructions + "\n" + user_prompt,
                    }
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
        ]
        llm_output = client.generate(messages, sampling_params=sampling_params)
        for completion in llm_output:
            return completion.outputs[0].text

    elif "gemini" in model_name:
        contents = user_prompt
        system_instruction = system_instructions
        response = client.models.generate_content(
            model=model_name,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=TEMPERATURE.value,
                # max_output_tokens=max_tokens,
                thinking_config=genai.types.ThinkingConfig(thinking_budget=-1), # if we want to control thinking; -1 is dynamic thinking
                top_p=1.0,
            ),
            contents=contents,
        )
        # print("Response:", response.text)  # Debugging output
        return response.text

    elif "gpt" in model_name: # GPT
        messages = [
            {
                "role": "system",
                "content": system_instructions,
            },
        ]
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            },
        )
        model = model_names.get(model_name, model_name)
        if "gpt-5" in model:
            # see https://platform.openai.com/docs/guides/reasoning
            # OpenAI recommends reserving at least 25000 tokens for reasoning and outputs.
            response = client.responses.create(
                model=model,
                reasoning={"effort": reasoning_effort},
                input=messages,
                max_output_tokens=25000
            )
            return response.output_text
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE.value,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return response.choices[0].message.content

    else:
        raise ValueError(f"Unknown model name: {model_name}. Please check the model name.")


def client() -> LLMClient:
    """Set up the LLM client.

    Returns:
      The LLM client depending on the name of the model.
    """
    if "claude" in MODEL.value:
        config = Config(read_timeout=1000)
        return boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            config=config,
        )
    elif MODEL.value == "qwen":
        return LLM(model=QWEN_PATH, quantization="gptq_marlin")
    elif MODEL.value == "llama":
        return LLM(
            model=LLAMA_PATH,
            dtype=torch.bfloat16,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            max_model_len=65536,
        )
    elif "gemini" in MODEL.value:
        return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    elif "gpt" in MODEL.value:  # GPT
        if OPEN_AI_API_KEY.value:
            return OpenAI(api_key=OPEN_AI_API_KEY.value)
        else: # try the environment variable; if not set, will raise an error
            # to set, `conda env config vars set OPENAI_API_KEY=your_secret_key` on conda.
            # or `export OPENAI_API_KEY=your_secret`_KEY` in bash.
            # You can check your env vars with `conda env config vars list`.
            return OpenAI()
