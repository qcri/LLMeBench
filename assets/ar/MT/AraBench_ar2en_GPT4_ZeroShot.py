from llmebench.datasets import AraBenchDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import MachineTranslationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
    }


def config():
    return {
        "dataset": AraBenchDataset,
        "dataset_args": {
            "src_lang": "ar",
            "tgt_lang": "en",
        },
        "task": MachineTranslationTask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 5,
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are an expert translator specialized in translating texts from Arabic to English. You are concise as you only output the translation of the text without any illustrations or extra details",
        },
        {
            "role": "user",
            "content": f"Translate the following text to English.\nText: {input_sample}\nTranslation: ",
        },
    ]


def post_process(response):
    response = response["choices"][0]["message"]["content"]
    response = response.replace('"', "")
    response = response.strip()
    return response
