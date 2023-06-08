import logging
import sys

import traceback
from abc import ABC, abstractmethod

from pathlib import Path

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


def log_retry(retry_state):
    if retry_state.attempt_number == 1:
        return
    logging.warning(f"Request failed, retry attempt {retry_state.attempt_number}...")


class ModelBase(object):
    def __init__(self, max_tries=5, retry_exceptions=(), **kwargs):
        self.max_tries = max_tries

        # Instantiate retrying mechanism
        self.prompt = retry(
            wait=wait_random_exponential(multiplier=1, max=60),
            stop=stop_after_attempt(self.max_tries),
            retry=retry_if_exception_type(retry_exceptions),
            before=log_retry,
            reraise=True,
        )(self.prompt)

    @abstractmethod
    def prompt(self, **kwargs):
        pass

    @abstractmethod
    def summarize_response(self, response):
        pass

    def run_model(self, processed_input):
        try:
            response = self.prompt(processed_input)
            return {"response": response}
        except Exception as e:
            exc_info = sys.exc_info()
            exception_str = "".join(traceback.format_exception(*exc_info))
            return {"failure_exception": exception_str}
