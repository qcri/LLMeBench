from llmebench.datasets import AraBenchDataset
from llmebench.models import HuggingFaceInferenceAPIModel, HuggingFaceTaskTypes
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
        "model": HuggingFaceInferenceAPIModel,
        "model_args": {
            "task_type": HuggingFaceTaskTypes.Translation,
            "inference_api_url": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-ar-en",
            "max_tries": 5,
        },
    }


def prompt(input_sample):
    return {"inputs": input_sample}


def post_process(response):
    return response[0]["translation_text"]
