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
        "scores": {"Micro-F1": ""},
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
    t_text = input_sample["text"]
    prompt = (
        "You are an expert social media image analyzer specializing in identifying propaganda in Arabic contexts. "
        "I will provide you with Arabic memes and the text extracted from these images. Your task is to briefly analyze them. "
        "To accurately perform this task, you will: (a) Explicitly focus on the image content to understand the context and provide a meaningful description and "
        "(b) pay close attention to the extracted text to enrich your description and support your analysis. "
        'Finally, provide response in valid JSON format with two fields with a format: {"description": "text", "classification": "propaganda"}. Output only json. '
        'The "description" should be very short in maximum 100 words and "classification" label should be "propaganda" or "not-propaganda" or "not-meme" or "other". '
        "Note, other is a category, which is used to label the image that does not fall in any of the previous category."
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
                {
                    "type": "text",
                    "text": f"Text extracted: {t_text}",
                },
            ],
        }
    ]


def post_process(response):
    data = response["choices"][0]["message"]["content"]
    data = re.search(r"```json\n(.*)\n```", data, re.DOTALL).group(1)
    data = json.loads(data)

    # Extract the classification label
    classification_label = data["classification"]

    return classification_label
