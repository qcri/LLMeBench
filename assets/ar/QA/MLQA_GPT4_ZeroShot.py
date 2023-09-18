from llmebench.datasets import MLQADataset
from llmebench.models import OpenAIModel
from llmebench.tasks import QATask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"F1": "0.620"},
    }


def config():
    return {
        "dataset": MLQADataset,
        "task": QATask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 50,
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "Assistant is a large language model trained by OpenAI.",
        },
        {
            "role": "user",
            "content": f"Your task is to answer questions in Arabic based on a given context.\nNote: Your answers should be spans extracted from the given context without any illustrations.\nYou don't need to provide a complete answer\nContext:{input_sample['context']}\nQuestion:{input_sample['question']}\nAnswer:",
        },
    ]


def post_process(response):
    return response["choices"][0]["message"]["content"]
