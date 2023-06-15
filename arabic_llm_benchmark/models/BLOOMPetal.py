from websockets.sync.client import connect
import json

from arabic_llm_benchmark.models.model_base import ModelBase


class BLOOMPetalFailure(Exception):
    def __init__(self, failure_type, failure_message):
        self.type_mapping = {
            "processing": "Model Inference failure",
            "connection": "Failed to connect to BLOOM Petal server"
        }
        self.type = failure_type
        self.failure_message = failure_message

    def __str__(self):
        return f"{self.type_mapping.get(self.type, self.type)}: \n {self.failure_message}"


class BLOOMPetalModel(ModelBase):
    def __init__(
        self, api_url, timeout=20, temperature=1e-7, top_p=0.95, max_tokens=800, **kwargs
    ):
        # API parameters
        self.api_url = api_url
        self.api_timeout = timeout
        self.request_header = {"type": "open_inference_session", "max_length": 1024}

        # GPT parameters
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        super(BLOOMPetalModel, self).__init__(
            retry_exceptions=(TimeoutError, BLOOMPetalFailure), **kwargs
        )

    def summarize_response(self, response):
        if (
            "outputs" in response
        ):
            return response['outputs']

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
                    raise BLOOMPetalFailure("processing", response['traceback'])    
            else:
                raise BLOOMPetalFailure(connect_message['traceback'])

        return response
