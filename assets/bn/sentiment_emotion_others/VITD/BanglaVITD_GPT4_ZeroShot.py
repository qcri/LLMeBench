from llmebench.datasets import BanglaVITDDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SentimentTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-3.5-turbo",
        "description": "GPT3.5 Turbo 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
    }


def config():
    return {
        "dataset": BanglaVITDDataset,
        "task": SentimentTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["Direct Violence", "Passive Violence", "Non-Violence"],
            "max_tries": 20,
        },
    }


def prompt(input_sample):
    prompt_string = f"""Based on the content of the text, please classify it as either "Direct Violence", "Passive Violence", or "Non-Violence". Provide only the label as your response. 

        text: {input_sample}

        label: """

    return [
        {
            "role": "system",
            "content": "You are a expert annotator. Your task is to analyze the text and identify the appropriate category of the text.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    if not response:
        return None
    label = response["choices"][0]["message"]["content"]

    label_fixed = label.replace("label:", "").strip()
    if label_fixed.startswith("Please provide the text"):
        label_fixed = None

    return label_fixed
