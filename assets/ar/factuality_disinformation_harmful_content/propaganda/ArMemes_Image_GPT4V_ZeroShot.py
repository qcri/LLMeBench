import json
import re

from llmebench.datasets import ArMemesDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ClassificationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-v",
        "description": "GPT4-o model hosted on Azure, using the ChatCompletion API. API version '2024-02-15-preview'.",
        "scores": {
            "Micro-F1": "0.5652557319223986",
            "Weighted F1": "0.5448863418109468",
            "Macro F1": "0.22338252349352886",
        },
    }


def config():
    return {
        "dataset": ArMemesDataset,
        "task": ClassificationTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["not_propaganda", "propaganda", "not_meme", "other"],
            "max_tries": 30,
        },
    }


def prompt(input_sample):
    base64_image = input_sample["image"]

    prompt = (
        "You are an expert social media image analyzer specializing in identifying propaganda in Arabic contexts. "
        "I will provide you with Arabic memes. "
        "Your task is to briefly analyze the image to answer questions. "
        'Finally, provide your response in valid JSON format with the following structure: {"classification": label}. '
        'The "classification" label must be one of the following: {"propaganda", "not_propaganda", "not-meme", or "other"}. '
        'Note: "other" is a category used to label the image if it cannot be classified as "propaganda", "not_propaganda", or "not-meme". '
        "Make sure that your response is a valid JSON."
    )

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ]


def post_process(response):
    data = response["choices"][0]["message"]["content"]
    data = json.loads(data)

    # Extract the classification label
    classification_label = data["classification"]

    return classification_label
