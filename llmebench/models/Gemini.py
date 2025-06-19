import json
import logging
import os

import requests

import vertexai
import vertexai.preview.generative_models as generative_models
from vertexai.generative_models import FinishReason, GenerativeModel, Part

from llmebench.models.model_base import ModelBase


class GeminiFailure(Exception):
    """Exception class to map various failure types from the Gemini server"""

    def __init__(self, failure_type, failure_message):
        self.type_mapping = {
            "processing": "Model Inference failure",
            "connection": "Failed to connect to Google Server",
        }
        self.type = failure_type
        self.failure_message = failure_message

    def __str__(self):
        return (
            f"{self.type_mapping.get(self.type, self.type)}: \n {self.failure_message}"
        )


class GeminiModel(ModelBase):
    """
    Gemini Model interface.

    Arguments
    ---------
    project_id : str
        Google Project ID. If not provided, the implementation will
        look at environment variable `GOOGLE_PROJECT_ID`
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
        project_id=None,
        api_key=None,
        model_name=None,
        timeout=20,
        temperature=0,
        top_p=0.95,
        max_tokens=2000,
        **kwargs,
    ):
        # API parameters
        # self.api_url = api_url or os.getenv("AZURE_DEPLOYMENT_API_URL")
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.project_id = project_id or os.getenv("GOOGLE_PROJECT_ID")
        self.model_name = model_name or os.getenv("MODEL")
        if self.api_key is None:
            raise Exception(
                "API Key must be provided as model config or environment variable (`GOOGLE_API_KEY`)"
            )
        if self.project_id is None:
            raise Exception(
                "PROJECT_ID must be provided as model config or environment variable (`GOOGLE_PROJECT_ID`)"
            )
        self.api_timeout = timeout
        self.safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
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

        super(GeminiModel, self).__init__(
            retry_exceptions=(TimeoutError, GeminiFailure), **kwargs
        )
        vertexai.init(project=self.project_id, location="us-central1")
        # self.client = GenerativeModel(self.model_name)

    def summarize_response(self, response):
        """Returns the "outputs" key's value, if available"""
        if "messages" in response:
            return response["messages"]

        return response

    def prompt(self, processed_input):
        """
        Gemini API Implementation

        Arguments
        ---------
        processed_input : list
            Must be list of dictionaries, where each dictionary has two keys;
            "role" defines a role in the chat (e.g. "system", "user") and
            "content" defines the actual message for that turn

        Returns
        -------
        response : Gemini API response
            Response from the Gemini server

        Raises
        ------
        GeminiFailure : Exception
            This method raises this exception if the server responded with a non-ok
            response
        """
        # headers = {
        #     "Content-Type": "application/json",
        #     "Authorization": "Bearer " + self.api_key,
        # }
        # body = {
        #     "input_data": {
        #         "input_string": processed_input,
        #         "parameters": {
        #             "max_tokens": self.max_tokens,
        #             "temperature": self.temperature,
        #             "top_p": self.top_p,
        #         },
        #     }
        # }
        generation_config = {
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        try:
            client = GenerativeModel(
                self.model_name, system_instruction=[processed_input[0]["content"]]
            )
            response = client.generate_content(
                [processed_input[1]["content"]],
                generation_config=generation_config,
                safety_settings=self.safety_settings,
            )

        except Exception as e:
            raise GeminiFailure(
                "processing",
                "Processing failed with status: {}".format(e),
            )

        # Parse the final response
        try:
            # response_data = response.json()
            response_data = [response.to_dict() for response in response.candidates]
        except Exception as e:
            raise GeminiFailure(
                "processing",
                "Processing failed: {}".format(response),
            )

        return response_data
