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

    Implementations of this class need to define at least two mandatory methods;
    `prompt()` and `summarize_response()`. Implementations of this class should target
    a specific model inference API, such as a platform (Azure, OpenAI), custom
    hosted inference server (Petals, FastChat) or other model-specific APIs.

    Attributes
    ----------
    max_tries : int, defaults to 5
        Defines how many retries are allowed per-sample in case of failure.
        Failure is defined by `retry_exceptions`.
    retry_exceptions : tuple
        Tuple of exceptions on which the framework should retry the request
        for any given sample. Specific exceptions should be included by the
        implementing class, such as HTTP Request failures (in case of HTTP-
        based APIs).

    Methods
    -------
    prompt(processed_input):
        Method that takes inputs from an asset and makes the actual request
        to the underlying model inference API.

    summarize_response(response):
        Method that takes a model response and summarizes it into a simpler
        form

    run_model(processed_input):
        Wrapper that calls the `prompt` method and captures exceptions
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
    def prompt(self, processed_input):
        """
        Method that implements communication to the underlying model

        Arguments
        ---------
        processed_input : dict
            Input from an asset. The structure of this will be dependent
            on a specific model implementation, and must be documented by
            the class implementation itself

        Returns
        -------
        response : mixed
            Response form the underlying model API

        Notes
        -----
        Ideally, this method will never be called directly, but through the
        `run_model` wrapper which takes care of returning the output in a
        consistent manner and also handles errors/exceptions.
        """
        pass

    @abstractmethod
    def summarize_response(self, response):
        """
        Method that summarizes/simplifies a model's response

        Arguments
        ---------
        response : mixed
            Response from `prompt()`

        Returns
        -------
        simplified_response : mixed
            Should ideally be a short string that summarizes the model's response
            (e.g. only the actual label instead of scores and other metadata). Will
            be saved in the summary file for quick debugging. If the response is not
            simplifiable, return the response object as is.
        """
        pass

    def run_model(self, processed_input):
        """
        Wrapper that calls the `prompt` method and captures exceptions

        Arguments
        ---------
        processed_input : dict
            Input from an asset. The structure of this will be dependent
            on a specific model implementation, and must be documented by
            the class implementation itself

        Returns
        -------
        response : dict
            Returns a dictionary with the key "response" holding the model's
            response, or "failure_exception" with the error that occurred when
            using the model
        """
        try:
            response = self.prompt(processed_input)
            return {"response": response}
        except Exception as e:
            exc_info = sys.exc_info()
            exception_str = "".join(traceback.format_exception(*exc_info))
            return {"failure_exception": exception_str}
