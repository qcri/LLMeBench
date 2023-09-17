import json
import os
import time

from enum import Enum

import requests

from llmebench.models.model_base import ModelBase

HuggingFaceTaskTypes = Enum(
    "HuggingFaceTaskTypes",
    [
        "Summarization",
        "Sentence_Similarity",
        "Text_Generation",
        "Text2Text_Generation",
        "Translation",
        "Feature_Extraction",
        "Fill_Mask",
        "Question_Answering",
        "Table_Question_Answering",
        "Text_Classification",
        "Token_Classification",
        "Named_Entity_Recognition",
        "Zero_Shot_Classification",
        "Conversational",
    ],
)


class HuggingFaceModelLoadingError(Exception):
    """Exception class to capture loading errors"""

    def __init__(self, failure_message):
        self.failure_message = failure_message

    def __str__(self):
        return f"HuggingFace model loading -- \n {self.failure_message}"


class HuggingFaceInferenceAPIModel(ModelBase):
    """
    An interface to HuggingFace Inference API

    Arguments
    ---------
    task_type : HuggingFaceTaskTypes
        One of Summarization, Sentence_Similarity, Text_Generation, Text2Text_Generation, Translation,
        Feature_Extraction, Fill_Mask, Question_Answering, Table_Question_Answering, Text_Classification,
        Token_Classification, Named_Entity_Recognition, Zero_Shot_Classification, Conversational as found on
        HuggingFace model's page
    inference_api_url : str
        The URL to the particular model, as found in the Deploy > Inference API menu in the model's page
    api_token : str
        HuggingFace API access key. If not provided, will be inferred from the environment variable
        `HUGGINGFACE_API_TOKEN`
    """

    def __init__(self, task_type, inference_api_url, api_token=None, **kwargs):
        self.task_type = task_type
        self.inference_api_url = inference_api_url
        self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN")

        if self.api_token is None:
            raise Exception(
                "API token must be provided as model config or environment variable (`HUGGINGFACE_API_TOKEN`)"
            )

        super(HuggingFaceInferenceAPIModel, self).__init__(
            retry_exceptions=(TimeoutError, HuggingFaceModelLoadingError), **kwargs
        )

    def prompt(self, processed_input):
        """
        HuggingFace Inference API Implementation

        Arguments
        ---------
        processed_input : dictionary
            Must be a dictionary with one key "inputs", the value of which will
            depend on the task type. See https://huggingface.co/docs/api-inference/detailed_parameters
            for detailed parameters.

        Returns
        -------
        response : dict
            Response from the HuggingFace Inference API

        Raises
        ------
        HuggingFaceModelLoadingError : Exception
            This method raises this exception if the model is not yet loaded on
            HuggingFace. Retrying after a few seconds is the usual remedy.
        """
        headers = {"Authorization": f"Bearer {self.api_token}"}
        data = json.dumps(processed_input)
        response = requests.request(
            "POST", self.inference_api_url, headers=headers, data=data
        )
        if not response.ok:
            if response.status_code == 503:  # model loading
                raise HuggingFaceModelLoadingError(response.reason)
            else:
                raise Exception(response.reason)
        return response.json()

    def summarize_response(self, response):
        """
        This method will attempt to interpret the output based on the task type.
        Otherwise, it returns the response object as is.
        """
        output_types = {
            HuggingFaceTaskTypes.Summarization: str,
            HuggingFaceTaskTypes.Sentence_Similarity: list,
            HuggingFaceTaskTypes.Text_Generation: str,
            HuggingFaceTaskTypes.Text2Text_Generation: str,
            HuggingFaceTaskTypes.Feature_Extraction: list,
        }
        output_dict_summary_keys = {
            HuggingFaceTaskTypes.Fill_Mask: ["token_str"],
            HuggingFaceTaskTypes.Question_Answering: ["answer"],
            HuggingFaceTaskTypes.Table_Question_Answering: ["answer"],
            HuggingFaceTaskTypes.Text_Classification: ["label", "score"],
            HuggingFaceTaskTypes.Token_Classification: ["entity_group", "word"],
            HuggingFaceTaskTypes.Named_Entity_Recognition: ["entity_group", "word"],
            HuggingFaceTaskTypes.Zero_Shot_Classification: ["scores"],
            HuggingFaceTaskTypes.Conversational: ["generated_text"],
            HuggingFaceTaskTypes.Translation: ["translation_text"],
        }

        output_type = output_types.get(self.task_type, dict)

        try:
            if output_type == list:
                return ", ".join([str(s) for s in response])

            if isinstance(response, list) and len(response) == 1:
                response = response[0]

            if output_type == dict:
                keys = output_dict_summary_keys[self.task_type]
                if isinstance(response, list):  # list of dictionaries
                    return ", ".join(
                        [":".join([str(d[key]) for key in keys]) for d in response]
                    )
                else:
                    return ":".join([str(response[key]) for key in keys])
            else:
                return response
        except Exception:
            return response
