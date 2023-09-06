import os

import openai

from llmebench.models.model_base import ModelBase


class OpenAIModelBase(ModelBase):
    def __init__(
        self,
        api_type=None,
        api_base=None,
        api_version=None,
        api_key=None,
        engine_name=None,
        model_name=None,
        temperature=0,
        top_p=0.95,
        max_tokens=800,
        frequency_penalty=0,
        presence_penalty=0,
        **kwargs
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

        openai.api_type = api_type

        if api_base:
            openai.api_base = api_base

        if api_version:
            openai.api_version = api_version

        if api_key:
            openai.api_key = api_key
        if openai.api_key is None:
            raise Exception(
                "API Key must be provided as model config or environment variable (`OPENAI_API_KEY` or `AZURE_API_KEY`)"
            )

        self.model_params = {}

        if model_name is None:
            raise Exception(
                "Model/Engine must be provided as model config or enviroment variable `OPENAI_MODEL`/`AZURE_ENGINE_NAME`"
            )

        if api_type == "azure":
            self.model_params["engine"] = model_name
        else:
            self.model_params["model"] = model_name

        # GPT parameters
        self.model_params["temperature"] = temperature
        self.model_params["top_p"] = top_p
        self.model_params["max_tokens"] = max_tokens
        self.model_params["frequency_penalty"] = frequency_penalty
        self.model_params["presence_penalty"] = presence_penalty

        super(OpenAIModelBase, self).__init__(
            retry_exceptions=(openai.error.Timeout, openai.error.RateLimitError),
            **kwargs
        )

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
            "api_key": os.getenv("OPEN_API_KEY"),
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
        if (
            "choices" in response
            and isinstance(response["choices"], list)
            and len(response["choices"]) > 0
            and "text" in response["choices"][0]
        ):
            return response["choices"][0]["text"]

        return None

    def prompt(self, processed_input):
        system_message = processed_input["system_message"]
        messages = processed_input["messages"]
        prompt = self.create_prompt(system_message, messages)
        response = openai.Completion.create(
            prompt=prompt, stop=["<|im_end|>"], **self.model_params
        )

        return response


class OpenAIModel(OpenAIModelBase):
    def summarize_response(self, response):
        if (
            "choices" in response
            and isinstance(response["choices"], list)
            and len(response["choices"]) > 0
            and "message" in response["choices"][0]
            and "content" in response["choices"][0]["message"]
            and response["choices"][0]["message"]["role"] == "assistant"
        ):
            return response["choices"][0]["message"]["content"]

        return None

    def prompt(self, processed_input):
        response = openai.ChatCompletion.create(
            messages=processed_input, **self.model_params
        )

        return response
