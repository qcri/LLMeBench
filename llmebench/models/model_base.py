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
    """
    Base class for models

    Implementors of this class need to implement at least two mandatory methods;
    `prompt()` and `summarize_response()`. Implementors of this class should target
    a specific model inference API, such as a platform (Azure, OpenAI), custom
    hosted inference server (Petals, FastChat) or other model-specific APIs.

    Attributes
    ----------
    max_tries : int, defaults to 5
        Defines how many retries are allowed per-sample in case of failure.
        Failure is defined by `retry_exceptions`.
    retru_exceptions : tuple
        Tuple of exceptions on which the framework should retry the request
        for any given sample. Specific exceptions should be included by the
        implementing class, such as HTTP Request failures (in case of HTTP-
        based APIs).

    Methods
    -------
    prompt(**kwargs):
        Method that takes inputs from an asset and makes the actual request
        to the underlying model inference API.

    summarize_response(response):
        Method that takes a model response and summarizes it into a simpler
        form
    """

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
