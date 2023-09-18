import os

from llmebench.datasets import ArSASDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SentimentTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Macro-F1": "0.598"},
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


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    for example in examples:
        out_prompt = (
            out_prompt
            + "Sentence: "
            + example["input"]
            + "\n"
            + example["label"]
            + "\n\n"
        )
    out_prompt = out_prompt + "Sentence: " + input_sample + "\n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = "Choose only one sentiment between: Positive, Negative, Neutral, or Mixed for this Arabic sentence."

    return [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    out = response["choices"][0]["message"]["content"]
    j = out.find(".")
    if j > 0:
        out = out[0:j]
    return out
