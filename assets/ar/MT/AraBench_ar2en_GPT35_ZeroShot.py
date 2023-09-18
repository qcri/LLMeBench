from llmebench.datasets import AraBenchDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import MachineTranslationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-35-turbo (version 0301)",
        "description": "GPT35 model hosted on Azure, using the Completion API. API version '2023-03-15-preview'.",
    }


def config():
    return {
        "dataset": AraBenchDataset,
        "dataset_args": {
            "src_lang": "ar",
            "tgt_lang": "en",
        },
        "task": MachineTranslationTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "max_tries": 5,
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an expert translator specialized in translating texts from Arabic to English. You are concise as you only output the translation of the text without any illustrations or extra details",
        "messages": [
            {
                "sender": "user",
                "text": f"Translate the following text to English.\nText: {input_sample}\nTranslation: ",
            }
        ],
    }


def post_process(response):
    response = response["choices"][0]["text"]
    response = response.replace('"', "")
    response = response.strip()
    return response
