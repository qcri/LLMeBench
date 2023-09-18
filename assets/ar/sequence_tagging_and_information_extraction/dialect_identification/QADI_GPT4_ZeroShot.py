from llmebench.datasets import QADIDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import DialectIDTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Macro-F1": "0.243"},
    }


def config():
    return {
        "dataset": QADIDataset,
        "dataset_args": {},
        "task": DialectIDTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "class_labels": [
                "EG",
                "DZ",
                "SD",
                "YE",
                "SY",
                "AE",
                "JO",
                "LY",
                "PS",
                "OM",
                "QA",
                "BH",
                "MSA",
                "SA",
                "IQ",
                "MA",
            ],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": f"Write only the country code of the Arabic country in which this sentence is written in its dialect without any explanation. Write only the country code in ISO 3166-1 alpha-2 format without explanation. Write 'MSA' if the sentence is written in Modern Standard Arabic.\n {input_sample}",
        },
    ]


def post_process(response):
    out = response["choices"][0]["message"]["content"]
    j = out.find(".")
    if j > 0:
        out = out[0:j]
    return out
