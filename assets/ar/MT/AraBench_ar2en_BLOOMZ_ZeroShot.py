from llmebench.datasets import AraBenchDataset
from llmebench.models import PetalsModel
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
        "model": PetalsModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return {
        "prompt": f"Translate from Arabic into English: {input_sample} \n",
    }


def post_process(response):
    return response["outputs"].strip().replace("<s>", "").replace("</s>", "")
