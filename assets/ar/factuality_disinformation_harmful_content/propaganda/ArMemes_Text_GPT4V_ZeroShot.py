import json

from llmebench.datasets import ArMemesDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ClassificationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-v",
        "description": "GPT-4o model hosted on Azure, using the ChatCompletion API. API version '2024-02-15-preview'.",
        "scores": {
            "Micro-F1": "0.6640211640211641",
            "Weighted F1": "0.6240374880108256",
            "Macro F1": "0.3840381788861714",
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
    t_text = input_sample["text"]
    prompt = (
        "You are an expert social media analyst specializing in identifying propaganda in Arabic contexts. "
        "I will provide you with the text extracted from images/memes observed on various social media platforms. "
        "Your task is to briefly analyze the text to answer questions. "
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
                    "type": "text",
                    "text": f"Text extracted: {t_text}",
                },
            ],
        }
    ]


def post_process(response):
    data = response["choices"][0]["message"]["content"]
    data = json.loads(data)
    # Extract the classification label
    classification_label = data["classification"]
    if classification_label == "not_meme":
        classification_label = "not-meme"

    return classification_label
