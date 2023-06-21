import os

from arabic_llm_benchmark.datasets import AdultDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import AdultTask


def config():
    return {
        "dataset": AdultDataset,
        "dataset_args": {},
        "task": AdultTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["ADULT", "NOT_ADULT"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/adult/adult-test.tsv"
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are an AI assistant that is an expert on language that helps people detect adult content.",
        },
        {
            "role": "user",
            "content": f'Classify the following Arabic sentence as adult language (the language used in adult advertisement and porno advertisement) or not adult language without illustruation. In case of adult language, just write "ADULT" without explaination, and in case of not adult language, just write "NOT_ADULT" without explaination \n {input_sample}',
        },
    ]


def post_process(response):
    out = response["choices"][0]["message"]["content"]
    j = out.find(".")
    if j > 0:
        out = out[0:j]
    return out
