import json
import os

from websockets.sync.client import connect

from llmebench.models.model_base import ModelBase


class PetalsFailure(Exception):
    """Exception class to map various failure types from the Petals server"""

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


class PetalsModel(ModelBase):
    """
    Petals Model interface.

    Arguments
    ---------
    api_url : str
        URL where the petals server is hosted. If not provided, the implementation will
        look at environment variable `PETALS_API_URL`
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
        self.api_url = api_url or os.getenv("PETALS_API_URL")
        if self.api_url is None:
            raise Exception(
                "API url must be provided as model config or environment variable (`PETALS_API_URL`)"
            )
        self.api_timeout = timeout
        self.request_header = {
            "type": "open_inference_session",
            "max_length": max_tokens,
        }

        # BLOOM parameters
        tolerance = 1e-7
        self.temperature = temperature
        if self.temperature < tolerance:
            # Currently, the model inference fails if temperature
            # is exactly 0, so we nudge it slightly to work around
            # the issue
            self.temperature += tolerance
        self.top_p = top_p
        self.max_tokens = max_tokens

        super(PetalsModel, self).__init__(
            retry_exceptions=(TimeoutError, PetalsFailure), **kwargs
        )

    def summarize_response(self, response):
        """Returns the "outputs" key's value, if available"""
        if "outputs" in response:
            return response["outputs"]

        return response

    def prompt(self, processed_input):
        """
        Petals API Implementation

        Arguments
        ---------
        processed_input : dictionary
            Must be a dictionary with one key "prompt", the value of which
            must be a string.

        Returns
        -------
        response : Petals API response
            Response from the petals server

        Raises
        ------
        PetalsFailure : Exception
            This method raises this exception if the server responded with a non-ok
            response
        """
        with connect(self.api_url, close_timeout=self.api_timeout) as websocket:
            websocket.send(json.dumps(self.request_header))
            connect_message = json.loads(websocket.recv())

            if connect_message["ok"]:
                params = {
                    "type": "generate",
                    "inputs": processed_input["prompt"],
                    "max_new_tokens": self.max_tokens,
                    "do_sample": 1,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
                encoded_message = json.dumps(params, separators=(",", ":"))
                websocket.send(encoded_message)
                received_message = websocket.recv()
                response = json.loads(received_message)

                if not response["ok"]:
                    raise PetalsFailure("processing", response["traceback"])
            else:
                raise PetalsFailure("connection", connect_message["traceback"])

        return response
