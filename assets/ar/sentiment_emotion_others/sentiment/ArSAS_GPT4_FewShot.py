import os

from llmebench.datasets import ArSASDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SentimentTask


def config():
    return {
        "dataset": ArSASDataset,
        "dataset_args": {},
        "task": SentimentTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["Positive", "Negative", "Neutral", "Mixed"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/sentiment/ArSAS-test.txt",
            "fewshot": {
                "train_data_path": "data/sentiment_emotion_others/sentiment/ArSAS-train.txt",
            },
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