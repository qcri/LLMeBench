from llmebench.datasets import AraBenchDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import MachineTranslationTask


def config():
    return {
        "dataset": AraBenchDataset,
        "dataset_args": {
            "src_lang": "ar",
            "tgt_lang": "en",
        },
        "task": MachineTranslationTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 5,
        }
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
