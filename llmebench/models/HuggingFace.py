import logging
import json
import requests
from llmebench.models.model_base import ModelBase
import os

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

API_TOKEN = os.environ["HUGGINGFACE_API_TOKEN"]


def log_retry(retry_state):
    if retry_state.attempt_number == 1:
        return
    logging.warning(
        f"Request failed, retry attempt {retry_state.attempt_number}...")


class HuggingFace(ModelBase):
    def __init__(self, inference_api_url, max_tries=5, retry_exceptions=()):
        self.max_tries = max_tries

        self.inference_api_url = inference_api_url
        self.api_token = API_TOKEN

        # Instantiate retrying mechanism
        self.prompt = retry(
            wait=wait_random_exponential(multiplier=1, max=60),
            stop=stop_after_attempt(self.max_tries),
            retry=retry_if_exception_type(retry_exceptions),
            before=log_retry,
            reraise=True,
        )(self.prompt)

    def prompt(self, processed_input):
        headers = {"Authorization": f"Bearer {self.api_token}"}
        data = json.dumps(processed_input)
        response = requests.request(
            "POST", self.inference_api_url, headers=headers, data=data)
        return response.json()

    def summarize_response(self, response):
        return response
