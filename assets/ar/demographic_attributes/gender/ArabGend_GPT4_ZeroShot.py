from llmebench.datasets import ArabGendDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import DemographyGenderTask


def config():
    return {
        "dataset": ArabGendDataset,
        "dataset_args": {},
        "task": DemographyGenderTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["m", "f"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": f"If the following person name can be considered as male, write 'm' without explanation, and if it can be considered as female, write 'f' without explanation.\n {input_sample}",
        },
    ]


def post_process(response):
    return response["choices"][0]["message"]["content"]
