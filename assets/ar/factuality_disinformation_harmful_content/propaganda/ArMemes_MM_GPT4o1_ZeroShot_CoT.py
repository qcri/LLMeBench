import json
import re

from llmebench.datasets import ArMemesDataset
from llmebench.models import OpenAIO1Model
from llmebench.tasks import ClassificationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4o1",
        "description": "GPT-4o1 model hosted on Azure, using the ChatCompletion API. API version '2024-02-15-preview'.",
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
        "model": OpenAIO1Model,
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
        "7) Verify the consistency between the classified label (step 4) and the explanation (step 6). If there is a mismatch, review and adjust accordingly.\n"
        "8) Using all extracted information ('text', 'description', 'entity mentions', multimodal details, and classification label), determine if the meme is:\n"
        "   (a) hateful\n"
        "   (b) not-hateful\n"
        "   Provide a justification for your decision regarding hatefulness.\n\n"
        "Your response should be formatted as a valid JSON object with the following structure:\n"
        "{\n"
        '  "extracted_text": "text",\n'
        '  "description": "short description",\n'
        '  "entity_mentions": ["mention1", "mention2"],\n'
        '  "classification": "propaganda/not_propaganda/other/not-meme",\n'
        '  "classification_explanation": "explanation",\n'
        '  "judgement": score (1-10),\n'
        '  "hate_label": "hateful/not-hateful",\n'
        '  "hatefulness_justification": "explanation"\n'
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
            "response_format": {"type": "json"},
        }
    ]


def post_process(response):
    if response is None or response == "":
        return None
    data = response["choices"][0]["message"]["content"]

    if data is None or data == "":
        return None
    # data = re.search(r"```json\n(.*)\n```", data, re.DOTALL).group(1)
    data = json.loads(data)

    # Extract the classification label
    classification_label = data["classification"]

    return classification_label
