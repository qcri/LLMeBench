import json
import os

import openai
from openai import AzureOpenAI, OpenAI

from llmebench.models.model_base import ModelBase


class OpenAIModelBase(ModelBase):
    """
    OpenAI Model interface. Can be used for models hosted on both OpenAI's platform and
    on Azure.

    Arguments
    ---------
    api_type : str
        Must be one of "openai" or "azure". If not provided, the implementation will try
        to induce it from environment variables `OPEN_API_TYPE`, `AZURE_*` or default to
        "openai"
    api_base : str
        URL where the model is hosted. Can be left as None for models hosted on OpenAI's
        platform. If not provided, the implementation will look at environment variables
        `OPENAI_API_BASE` or `AZURE_API_URL`
    api_version : str
        Version of the API to use. If not provided, the implementation will derive it
        from environment variables `OPENAI_API_VERSION` or `AZURE_API_VERSION`. Must be
        left as None for models hosted on OpenAI's platform
    api_key : str
        Authentication token for the API. If not provided, the implementation will derive it
        from environment variables `OPENAI_API_KEY` or `AZURE_API_KEY`.
    model_name : str
        Name of the model to use. If not provided, the implementation will derive it from
        environment variables `OPENAI_MODEL` or `AZURE_ENGINE_NAME`
    engine_name : str
        Alternative for `model_name`
    temperature : float
        Temperature value to use for the model. Defaults to zero for reproducibility.
    top_p : float
        Top P value to use for the model. Defaults to 0.95
    max_tokens : int
        Maximum number of tokens to pass to the model. Defaults to 800
    frequency_penalty : float
        Frequency Penalty to use for the model.
    presence_penalty : float
        Presence Penalty to use for the model.
    """

    def __init__(
        self,
        api_type=None,
        api_base=None,
        api_version=None,
        api_key=None,
        model_name=None,
        engine_name=None,
        temperature=0,
        top_p=0.95,
        max_tokens=800,
        frequency_penalty=0,
        presence_penalty=0,
        **kwargs,
    ):
        # API parameters
        # Order of priority is:
        #   1. arguments to the constructor
        #   2. OPENAI_* env vars
        #   3. AZURE_* env vars
        azure_vars = self.read_azure_env_vars()
        openai_vars = self.read_openai_env_vars()

        api_type = (
            api_type or openai_vars["api_type"] or azure_vars["api_type"] or "openai"
        )
        api_base = api_base or openai_vars["api_base"] or azure_vars["api_base"]
        api_version = (
            api_version or openai_vars["api_version"] or azure_vars["api_version"]
        )
        api_key = api_key or openai_vars["api_key"] or azure_vars["api_key"]
        model_name = (
            model_name or engine_name or openai_vars["model"] or azure_vars["model"]
        )

        if api_type == "azure" and api_base is None:
            raise Exception(
                "API URL must be provided as model config or environment variable (`AZURE_API_BASE`)"
            )

        openai.api_type = api_type

        if api_type == "azure" and api_version is None:
            raise Exception(
                "API version must be provided as model config or environment variable (`AZURE_API_VERSION`)"
            )

        if api_version:
            openai.api_version = api_version

        if api_key is None:
            raise Exception(
                "API Key must be provided as model config or environment variable (`OPENAI_API_KEY` or `AZURE_API_KEY`)"
            )

        self.model_params = {}

        if model_name is None:
            raise Exception(
                "Model/Engine must be provided as model config or environment variable `OPENAI_MODEL`/`AZURE_ENGINE_NAME`"
            )

        openai.api_key = api_key
        self.model_params["model"] = model_name

        # GPT parameters
        self.model_params["temperature"] = temperature
        self.model_params["top_p"] = top_p
        self.model_params["max_tokens"] = max_tokens
        self.model_params["frequency_penalty"] = frequency_penalty
        self.model_params["presence_penalty"] = presence_penalty

        super(OpenAIModelBase, self).__init__(
            retry_exceptions=(openai.Timeout, openai.RateLimitError), **kwargs
        )

        if api_type == "azure":
            self.client = AzureOpenAI(
                api_version=api_version,
                api_key=api_key,
                base_url=f"{api_base}/openai/deployments/{model_name}/",
            )
        elif api_type == "openai":
            if not api_base:
                api_base = "https://api.openai.com/v1"
            self.client = OpenAI(base_url=api_base, api_key=api_key)
        else:
            raise Exception('API type must be one of "azure" or "openai"')

    @staticmethod
    def read_azure_env_vars():
        curr_api_type = None
        if "AZURE_ENGINE_NAME" in os.environ or "ENGINE_NAME" in os.environ:
            curr_api_type = "azure"
        return {
            "api_type": curr_api_type,
            "api_version": os.getenv("AZURE_API_VERSION"),
            "api_base": os.getenv("AZURE_API_URL"),
            "api_key": os.getenv("AZURE_API_KEY"),
            "model": os.getenv("AZURE_ENGINE_NAME", os.getenv("ENGINE_NAME")),
        }

    @staticmethod
    def read_openai_env_vars():
        return {
            "api_type": os.getenv("OPEN_API_TYPE"),
            "api_version": os.getenv("OPENAI_API_VERSION"),
            "api_base": os.getenv("OPENAI_API_BASE"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("OPENAI_MODEL"),
        }


class LegacyOpenAIModel(OpenAIModelBase):
    # defining a function to create the prompt from the system and user messages
    def create_prompt(self, system_message, messages):
        system_message_template = "<|im_start|>system\n{}\n<|im_end|>"
        message_template = "\n<|im_start|>{}\n{}\n<|im_end|>"

        prompt = system_message_template.format(system_message)

        for message in messages:
            prompt += message_template.format(message["sender"], message["text"])
        prompt += "\n<|im_start|>assistant\n"
        return prompt

    def summarize_response(self, response):
        """Returns the first reply, if available"""
        if (
            "choices" in response
            and isinstance(response.choices, list)
            and len(response.choices) > 0
            and "text" in response.choices[0]
        ):
            return response.choices[0].text

        return response

    def prompt(self, processed_input):
        """
        OpenAI API Completion implementation

        .. warning::
        This implementation is deprecated and will be removed in future versions. Use
        `OpenAIModel` instead.

        Arguments
        ---------
        processed_input : dict
            Must be a dictionary with two keys; "system_message" with a string
            value, and "messages" with a list value, where each element is a
            dictionary with two string-valued keys, "sender" and "text".

        Returns
        -------
        response : OpenAI API response
            Response from the openai python library
        """
        system_message = processed_input["system_message"]
        messages = processed_input["messages"]
        prompt = self.create_prompt(system_message, messages)
        response = self.client.completions.create(
            prompt=prompt, stop=["<|im_end|>"], **self.model_params
        )

        return response


class OpenAIModel(OpenAIModelBase):
    def summarize_response(self, response):
        """Returns the first reply from the "assistant", if available"""
        if (
            "choices" in response
            and isinstance(response["choices"], list)
            and len(response["choices"]) > 0
            and "message" in response["choices"][0]
            and "content" in response["choices"][0]["message"]
            and response["choices"][0]["message"]["role"] == "assistant"
        ):
            return response["choices"][0]["message"]["content"]

        return response

    def prompt(self, processed_input):
        """
        OpenAI API ChatCompletion implementation

        Arguments
        ---------
        processed_input : list
            Must be list of dictionaries, where each dictionary has two keys;
            "role" defines a role in the chat (e.g. "system", "user") and
            "content" defines the actual message for that turn

        Returns
        -------
        response : OpenAI API response
            Response from the openai python library

        """
        response = self.client.chat.completions.create(
            messages=processed_input, **self.model_params
        )
        response = json.loads(response.json())
        return response
