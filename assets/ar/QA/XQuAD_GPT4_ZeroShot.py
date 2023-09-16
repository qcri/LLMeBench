from llmebench.datasets import XQuADDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import QATask


def config():
    return {
        "dataset": XQuADDataset,
        "dataset_args": {},
        "task": QATask,
        "task_args": {},
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
