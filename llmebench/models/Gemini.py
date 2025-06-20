import json
import logging
import os

import requests

import vertexai
import vertexai.preview.generative_models as generative_models
from vertexai.generative_models import FinishReason, GenerativeModel, Part
from google.oauth2 import service_account
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
        model_name=None,
        location=None,
        credentials_path=None,      # path to JSON file
        credentials_info=None,      # dict or JSON string
        timeout=20,
        temperature=0,
        tolerance = 1e-7,
        top_p=0.95,
        max_tokens=2000,
        **kwargs,
    ):
        self.project_id = project_id or os.getenv("GOOGLE_PROJECT_ID")
        self.model_name = model_name or os.getenv("MODEL")
        self.location = location or os.getenv("VERTEX_LOCATION") or "us-central1"
        self.credentials = None

        # 1. Prefer explicit credentials_info (dict or JSON string)
        if credentials_info:
            if isinstance(credentials_info, str):
                credentials_info = json.loads(credentials_info)
            self.credentials = service_account.Credentials.from_service_account_info(credentials_info)
        # 2. Else, load from path (arg or env)
        elif credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            with open(path, "r") as f:
                info = json.load(f)
            self.credentials = service_account.Credentials.from_service_account_info(info)
        elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is not None:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        # 3. Else, None: will fall back to ADC (Application Default Credentials)

        if not self.project_id:
            raise Exception("PROJECT_ID must be set (argument or `GOOGLE_PROJECT_ID` in .env)")
        if not self.model_name:
            raise Exception("MODEL must be set (argument or `MODEL` in .env)")
        if not self.location:
            raise Exception("LOCATION must be set (argument or `VERTEX_LOCATION` in .env)")

        vertexai.init(
            project=self.project_id,
            location=self.location,
            credentials=self.credentials
        )

        self.tolerance = tolerance
        self.temperature = max(temperature, tolerance)
        self.top_p = top_p
        self.max_tokens = max_tokens

        self.safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        super(GeminiModel, self).__init__(
            retry_exceptions=(TimeoutError, GeminiFailure), **kwargs
        )

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
