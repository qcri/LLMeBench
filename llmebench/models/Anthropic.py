import json
import logging
import os

import anthropic

from llmebench.models.model_base import ModelBase


class AnthropicFailure(Exception):
    """Exception class to map various failure types from the AzureModel server"""

    def __init__(self, failure_type, failure_message):
        self.type_mapping = {
            "processing": "Model Inference failure",
            "connection": "Failed to connect to the API endpoint",
        }
        self.type = failure_type
        self.failure_message = failure_message

    def __str__(self):
        return (
            f"{self.type_mapping.get(self.type, self.type)}: \n {self.failure_message}"
        )


class AnthropicModel(ModelBase):
    """
    Anthropic Model interface.

    Arguments
    ---------
    api_url : EMPTY
    timeout : int
        Number of seconds before the request to the server is timed out
    temperature : float
        Temperature value to use for the model. Defaults to zero for reproducibility.
    top_p : float
        Top P value to use for the model. Defaults to 0.95
    max_tokens : int
        Maximum number of tokens to pass to the model. Defaults to 1512
    """

    def __init__(
        self,
        api_base=None,
        api_key=None,
        model_name=None,
        timeout=20,
        temperature=0,
        top_p=0.95,
        max_tokens=2000,
        **kwargs,
    ):
        # API parameters
        self.api_base = api_base or os.getenv("ANTHROPIC_API_URL")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model_name = model_name or os.getenv("ANTHROPIC_MODEL")

        # Parameters
        self.api_timeout = timeout
        tolerance = 1e-7
        self.temperature = temperature
        if self.temperature < tolerance:
            # Currently, the model inference fails if temperature
            # is exactly 0, so we nudge it slightly to work around
            # the issue
            self.temperature += tolerance
        self.top_p = top_p
        self.max_tokens = max_tokens

        if self.api_key is None:
            raise Exception(
                "API key must be provided as model config or environment variable (`ANTHROPIC_API_KEY`)"
            )
        if self.model_name is None:
            raise Exception(
                "Model name must be provided as model config or environment variable (`ANTHROPIC_MODEL`)"
            )
        self.model = self.model_name
        # GPT parameters
        self.model_params = {}
        self.model_params["system"] = (
            kwargs.get("system_msg")
            if "system_msg" in kwargs and kwargs["system_msg"]
            else "You are an expert AI assistant"
        )
        self.model_params["temperature"] = temperature
        self.model_params["top_p"] = top_p
        self.model_params["max_tokens"] = max_tokens
        self.client = anthropic.Anthropic(api_key=self.api_key)

        super(AnthropicModel, self).__init__(
            retry_exceptions=(TimeoutError, AnthropicFailure), **kwargs
        )

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
        AnthropicModel API Implementation

        Arguments
        ---------
        processed_input : dictionary
            Must be a dictionary with one key "prompt", the value of which
            must be a string.

        Returns
        -------
        response : AnthropicModel API response
        """

        response = self.client.messages.create(
            model=self.model, messages=processed_input, **self.model_params
        )
        response = json.loads(response.json())
        return response
