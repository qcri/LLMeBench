import json
import logging
import os

import anthropic

from llmebench.models.model_base import ModelBase


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
        api_key=None,
        model_name=None,
        timeout=20,
        temperature=0,
        top_p=0.95,
        max_tokens=2000,
        **kwargs,
    ):
        # API parameters
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
            retry_exceptions=(
                TimeoutError,
                anthropic.APIStatusError,
                anthropic.RateLimitError,
                anthropic.APITimeoutError,
                anthropic.APIConnectionError,
            ),
            **kwargs,
        )

    def summarize_response(self, response):
        """Returns the response"""

        return response

    def prompt(self, processed_input):
        """
        AnthropicModel API Implementation

        Arguments
        ---------
        processed_input : list
            Must be list of dictionaries, where each dictionary has two keys;
            "role" defines a role in the chat (e.g. "user") and
            "content" can be a list or message for that turn. If it is a list, it must contain objects matching one of the following:
                - {"type": "text", "text": "....."} for text input/prompt
                - {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "media_file"}} for image input
                - the list can contain mix of the above formats for multimodal input (image + text)

        Returns
        -------
        response : AnthropicModel API response
            Response from the anthropic python library

        """

        response = self.client.messages.create(
            model=self.model, messages=processed_input, **self.model_params
        )
        response = json.loads(response.json())
        return response
