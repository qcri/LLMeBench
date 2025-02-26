import json
import re

from llmebench.datasets import ArMemesDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ClassificationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4o",
        "description": "GPT-4o model hosted on Azure, using the ChatCompletion API. API version '2024-02-15-preview'.",
        "scores": {
            "Micro-F1": "0.5661375661375662",
            "Weighted F1": "0.5296411577439224",
            "Macro F1": "0.2514937653942827",
        },
    }


def config():
    return {
        "dataset": ArMemesDataset,
        "task": ClassificationTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["not_propaganda", "propaganda", "not-meme", "other"],
            "max_tries": 30,
            "max_tokens": 1000,
        },
    }


def prompt(input_sample):
    base64_image = input_sample["image"]

    prompt = (
        "You are an expert social media analyst specializing in identifying propaganda in Arabic contexts. "
        "I will provide you with an Arabic meme, and your task is to analyze it by following these steps:\n\n"
        "1) Extract the 'text' from the meme.\n"
        "2) Generate a concise 'description' of the image in no more than 50 words.\n"
        "3) Identify and extract 'entity mentions' (e.g., PERSON, ORGANIZATION, LOCATION) if present.\n"
        "4) Using the extracted 'text', 'description', 'entity mentions', and other multimodal information, classify the meme into one of the following categories:\n"
        "   (a) propaganda\n"
        "   (b) not_propaganda\n"
        "   (c) other (if it does not fit into either category above)\n"
        "   (d) not-meme (if it does not qualify as a meme)\n"
        "5) Assess the confidence of your classification on a scale from 1 to 10.\n"
        "6) Justify your classification decision with an explanation.\n"
        "Your response should be formatted as a valid JSON object with the following structure:\n"
        "{\n"
        '  "extracted_text": "text",\n'
        '  "description": "short description",\n'
        '  "entity_mentions": ["mention1", "mention2"],\n'
        '  "classification": "propaganda/not_propaganda/other/not-meme",\n'
        '  "classification_explanation": "explanation",\n'
        '  "judgement": score (1-10),\n'
        "}"
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
    data = re.search(r"```json\n(.*)\n```", data, re.DOTALL).group(1)
    data = json.loads(data)

    # Extract the classification label
    classification_label = data["classification"]

    return classification_label
