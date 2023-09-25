from llmebench.datasets import AraBenchDataset
from llmebench.models import FastChatModel
from llmebench.tasks import MachineTranslationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using FastChat.",
    }


def config():
    return {
        "dataset": AraBenchDataset,
        "dataset_args": {
            "src_lang": "ar",
            "tgt_lang": "en",
        },
        "task": MachineTranslationTask,
        "model": FastChatModel,
        "model_args": {
            "max_tries": 3,
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
