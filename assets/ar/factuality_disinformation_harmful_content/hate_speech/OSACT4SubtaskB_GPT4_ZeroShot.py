from llmebench.datasets import OSACT4SubtaskBDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import HateSpeechTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Macro-F1": "0.669"},
    }


def config():
    return {
        "dataset": OSACT4SubtaskBDataset,
        "task": HateSpeechTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["HS", "NOT_HS"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are an expert annotator, you can identify and label hate speech content within a tweet.",
        },
        {
            "role": "user",
            "content": f'Given the following tweet, label it as "HS" or "NOT_HS" based on the content of the tweet: \n {input_sample}',
        },
    ]


def post_process(response):
    out = response["choices"][0]["message"]["content"]
    j = out.find(".")
    if j > 0:
        out = out[0:j]

    if "not_hate_speech" in out or "no_hate_speech" in out or "NOT_HS" == out:
        out = "NOT_HS"
    elif "hate_speech" in out or "HS" == out:
        out = "HS"
    else:
        out = None
    return out
