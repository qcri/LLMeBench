import json

from websockets.sync.client import connect

from llmebench.models.model_base import ModelBase


class PetalsFailure(Exception):
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
    def __init__(
        self, api_url, timeout=20, temperature=0, top_p=0.95, max_tokens=1512, **kwargs
    ):
        # API parameters
        self.api_url = api_url
        self.api_timeout = timeout
        self.request_header = {"type": "open_inference_session", "max_length": 1512}

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
        if "outputs" in response:
            return response["outputs"]

        return None

    def prompt(self, processed_input):
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
                raise PetalsFailure(connect_message["traceback"])

        return response
