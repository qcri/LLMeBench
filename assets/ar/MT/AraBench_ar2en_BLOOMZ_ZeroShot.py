from llmebench.datasets import AraBenchDataset
from llmebench.models import PetalsModel
from llmebench.tasks import MachineTranslationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
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
