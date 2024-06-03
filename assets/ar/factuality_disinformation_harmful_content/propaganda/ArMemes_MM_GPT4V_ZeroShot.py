import json
import re

from llmebench.datasets import ArMemesDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ClassificationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-v",
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
            "class_labels": ["not_propaganda", "propaganda", "not_meme", "other"],
            "max_tries": 30,
        },
    }


def prompt(input_sample):
    base64_image = input_sample["image"]
    prompt = (
        "You are an expert social media analyst specializing in identifying propaganda in Arabic contexts. "
        "I will provide you with an Arabic meme. You need to perform the following tasks:\n"
        "1) Your task is to extract text from the meme.\n"
        "2) Extract description.\n"
        "3) Extract entity mentions if there are any. For example, PERSON.\n"
        "4) Use all this information to identify whether the meme is (a) propaganda, (b) not_propaganda, (c) other, or (d) not-meme.\n"
        "5) Then judge your decision on a scale of 1 to 10.\n"
        "6) Use all the information extracted in step 1, 2 and 3 to identify whether the meme is (a) 'hateful' or (b) 'not-hateful'\n"
        "Note: 'other' is a category used to label the image if it cannot be classified as 'propaganda', 'not_propaganda', or 'not-meme'.\n"
        "You should output all extracted information in valid JSON with the format below:\n"
        "{\n"
        '"extracted_text": "text",\n'
        '"description": "description",\n'
        '"entity_mentions": "mention1, mention2",\n'
        '"classification": "propagandistic",\n'
        '"judgement": score,\n'
        '"hate_label": hateful\n'
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
    print(data)
    data = re.search(r"```json\n(.*)\n```", data, re.DOTALL).group(1)
    data = json.loads(data)

    # Extract the classification label
    classification_label = data["classification"]

    return classification_label
