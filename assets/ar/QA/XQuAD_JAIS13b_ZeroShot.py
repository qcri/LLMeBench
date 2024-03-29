from llmebench.datasets import XQuADDataset
from llmebench.models import FastChatModel
from llmebench.tasks import QATask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Jais-13b-chat",
        "description": "Locally hosted Jais-13b-chat model using FastChat.",
        "scores": {"F1": "0.636"},
    }


def config():
    return {
        "dataset": XQuADDataset,
        "task": QATask,
        "model": FastChatModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    base_prompt = f"Your task is to answer questions in Arabic based on a given context.\nNote: Your answers should be spans extracted from the given context without any illustrations.\nYou don't need to provide a complete answer\nContext:{input_sample['context']}\nQuestion:{input_sample['question']}\nAnswer:"

    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    return response["choices"][0]["message"]["content"]
