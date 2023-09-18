from llmebench.datasets import ArSASDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SentimentTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Macro-F1": "0.569"},
    }


def config():
    return {
        "dataset": ArSASDataset,
        "task": SentimentTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["Positive", "Negative", "Neutral", "Mixed"],
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
            "content": f"Choose only one sentiment between: Positive, Negative, Neutral, or Mixed for this Arabic sentence: \n {input_sample}",
        },
    ]


def post_process(response):
    out = response["choices"][0]["message"]["content"]
    j = out.find(".")
    if j > 0:
        out = out[0:j]
    return out
