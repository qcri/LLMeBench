import os

from llmebench.datasets import BanglaSentimentDataset
from llmebench.models import GPTChatCompletionModel
from llmebench.tasks import SentimentTask


def config():
    return {
        "dataset": BanglaSentimentDataset,
        "dataset_args": {},
        "task": SentimentTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["Positive", "Negative", "Neutral"],
            "max_tries": 20,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/sentiment/bn/bn_all_test.tsv",
            "fewshot": {
                "train_data_path": "data/sentiment_emotion_others/sentiment/bn/bn_all_train.tsv",
            },
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    out_prompt = out_prompt + "Here are some examples:\n\n"

    for index, example in enumerate(examples):
        out_prompt = (
            out_prompt
            + "Example "
            + str(index)
            + ":"
            + "\n"
            + "text: "
            + example["input"]
            + "\nlabel: "
            + example["label"]
            + "\n\n"
        )

    out_prompt = out_prompt + "text: " + input_sample + "\nlabel: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = 'Annotate the "text" into "one" of the following categories: Positive, Negative or Neutral'
    return [
        {
            "role": "system",
            "content": f"You are a expert annotator. Your task is to analyze the text and identify sentiment polarity.\n",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label_fixed = label.replace("label:", "").strip()
    if label_fixed.startswith("Please provide the text"):
        label_fixed = None

    return label_fixed