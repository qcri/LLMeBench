import json
import logging
import os

import requests

from llmebench.models.model_base import ModelBase


class AzureModelFailure(Exception):
    """Exception class to map various failure types from the AzureModel server"""

    def __init__(self, failure_type, failure_message):
        self.type_mapping = {
            "processing": "Model Inference failure",
            "connection": "Failed to connect to Azure deployment",
        }
        self.type = failure_type
        self.failure_message = failure_message

    def __str__(self):
        return (
            f"{self.type_mapping.get(self.type, self.type)}: \n {self.failure_message}"
        )


class AzureModel(ModelBase):
    """
    AzureModel Model interface.

    Arguments
    ---------
    api_url : str
        URL where the AzureModel server is hosted. If not provided, the implementation will
        look at environment variable `AZURE_DEPLOYMENT_API_URL`
    api_key : str
        Authentication token for the API. If not provided, the implementation will derive it
        from environment variables `OPENAI_API_KEY` or `AZURE_API_KEY`.
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
        api_key=None,
        timeout=20,
        temperature=0,
        top_p=0.95,
        max_tokens=2000,
        **kwargs,
    ):
        # API parameters
        self.api_url = api_url or os.getenv("AZURE_DEPLOYMENT_API_URL")
        self.api_key = api_key or os.getenv("AZURE_DEPLOYMENT_API_KEY")
        if self.api_url is None:
            raise Exception(
                "API url must be provided as model config or environment variable (`AZURE_DEPLOYMENT_API_URL`)"
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

        super(AzureModel, self).__init__(
            retry_exceptions=(TimeoutError, AzureModelFailure), **kwargs
        )

    def summarize_response(self, response):
        """Returns the "outputs" key's value, if available"""
        if "messages" in response:
            return response["messages"]

        return response

    def prompt(self, processed_input):
        """
        AzureModel API Implementation

        Arguments
        ---------
        processed_input : list
            Must be list of dictionaries, where each dictionary has two keys;
            "role" defines a role in the chat (e.g. "system", "user") and
            "content" defines the actual message for that turn

        Returns
        -------
        response : AzureModel API response
            Response from the AzureModel server

        Raises
        ------
        AzureModelFailure : Exception
            This method raises this exception if the server responded with a non-ok
            response
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
        }
        body = {
            "input_data": {
                "input_string": processed_input,
                "parameters": {
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                },
            }
        }
        try:
            response = requests.post(
                self.api_url, headers=headers, json=body, timeout=self.api_timeout
            )

        except Exception as e:
            raise AzureModelFailure(
                "connection",
                f"Failed to connect to {self.api_url}",
            )

        if response.status_code != 200:
            raise AzureModelFailure(
                "processing",
                "Processing failed with status: {}".format(response.status_code),
            )

        # Parse the final response
        try:
            response_data = response.json()
        except Exception as e:
            raise AzureModelFailure(
                "processing",
                "Processing failed: {}".format(response),
            )

        return response_data
