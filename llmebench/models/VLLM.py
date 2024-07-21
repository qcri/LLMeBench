import json
import logging
import os

from llmebench.models.OpenAI import OpenAIModel


class VLLMModel(OpenAIModel):
    """
    VLLM Model interface. Can be used for models hosted using https://github.com/vllm-project/vllm.

    Accepts all arguments used by `OpenAIModel`, and overrides the arguments listed
    below with VLLM variables.

    See the [https://docs.vllm.ai/en/latest/models/supported_models.html](model_support)
    page in VLLM's documentation for supported models and instructions on extending
    to custom models.

    Arguments
    ---------
    api_url : str
        URL where the VLLM server is hosted. If not provided, the implementation will
        look at environment variable `VLLM_API_URL`
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
        max_tokens=1512,
        **kwargs,
    ):
        # API parameters
        self.api_base = api_base or os.getenv("VLLM_API_URL")
        self.api_key = api_key or os.getenv("VLLM_API_KEY")
        self.model_name = model_name or os.getenv("VLLM_MODEL")
        self.api_timeout = timeout
        # Parameters
        tolerance = 1e-7
        self.temperature = temperature
        if self.temperature < tolerance:
            # Currently, the model inference fails if temperature
            # is exactly 0, so we nudge it slightly to work around
            # the issue
            self.temperature += tolerance
        self.top_p = top_p
        self.max_tokens = max_tokens

        if self.api_base is None:
            raise Exception(
                "API url must be provided as model config or environment variable (`VLLM_API_BASE`)"
            )
        if self.api_key is None:
            raise Exception(
                "API key must be provided as model config or environment variable (`VLLM_API_KEY`)"
            )
        if self.model_name is None:
            raise Exception(
                "Model name must be provided as model config or environment variable (`VLLM_MODEL`)"
            )
        # checks for valid config settings)
        super(VLLMModel, self).__init__(
            api_base=self.api_base,
            api_key=self.api_key,
            model_name=self.model_name,
            **kwargs,
        )
