import json
import requests
import time
from llmebench.models.model_base import ModelBase


class HuggingFaceModelLoadingError(Exception):
    def __init__(self, failure_message):
        self.failure_message = failure_message

    def __str__(self):
        return (
            f"HuggingFace model loading -- \n {self.failure_message}"
        )


class HuggingFaceInferenceAPIModel(ModelBase):
    def __init__(self, inference_api_url, api_token, **kwargs):
        self.inference_api_url = inference_api_url
        self.api_token = api_token

        super(HuggingFaceInferenceAPIModel, self).__init__(
            retry_exceptions=(TimeoutError, HuggingFaceModelLoadingError), **kwargs)

    def prompt(self, processed_input):
        headers = {"Authorization": f"Bearer {self.api_token}"}
        data = json.dumps(processed_input)
        response = requests.request(
            "POST", self.inference_api_url, headers=headers, data=data)
        if not response.ok:
            if response.status_code == 503:  # model loading
                time.sleep(1)
                raise HuggingFaceModelLoadingError(
                    response.reason)
            else:
                raise Exception(response.reason)
        return response.json()

    def summarize_response(self, response):
        return response
