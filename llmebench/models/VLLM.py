import json
import logging
import os

import requests

from llmebench.models.model_base import ModelBase


class VLLMFailure(Exception):
    """Exception class to map various failure types from the VLLM server"""

    def __init__(self, failure_type, failure_message):
        self.type_mapping = {
            "processing": "Model Inference failure",
            "connection": "Failed to connect to BLOOM Petal server",
        }
        self.type = failure_type
        self.failure_message = failure_message

    def __str__(self):
        return (
            f"{self.type_mapping.get(self.type, self.type)}: \n {self.failure_message}"
        )


class VLLMModel(ModelBase):
    """
    VLLM Model interface.

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
        api_url=None,
        timeout=20,
        temperature=0,
        top_p=0.95,
        max_tokens=1512,
        **kwargs,
    ):
        # API parameters
        self.api_url = api_url or os.getenv("VLLM_API_URL")
        self.user_session_id = os.getenv("USER_SESSION_ID")
        if self.api_url is None:
            raise Exception(
                "API url must be provided as model config or environment variable (`VLLM_API_URL`)"
            )
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

        super(VLLMModel, self).__init__(
            retry_exceptions=(TimeoutError, VLLMFailure), **kwargs
        )

    def summarize_response(self, response):
        """Returns the "outputs" key's value, if available"""
        if "messages" in response:
            return response["messages"]

        return response

    def prompt(self, processed_input):
        """
        VLLM API Implementation

        Arguments
        ---------
        processed_input : dictionary
            Must be a dictionary with one key "prompt", the value of which
            must be a string.

        Returns
        -------
        response : VLLM API response
            Response from the VLLM server

        Raises
        ------
        VLLMFailure : Exception
            This method raises this exception if the server responded with a non-ok
            response
        """
        headers = {"Content-Type": "application/json"}
        params = {
            "messages": processed_input,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "user_session_id": self.user_session_id,
        }
        try:
            response = requests.post(
                self.api_url, headers=headers, json=params, timeout=self.api_timeout
            )
            if response.status_code != 200:
                raise VLLMFailure(
                    "processing",
                    "Processing failed with status: {}".format(response.status_code),
                )

            # Parse the final response
            response_data = response.json()
            logging.info(f"initial_response: {response_data}")
        except VLLMFailure as e:
            print("Error occurred:", e)

        return response_data
