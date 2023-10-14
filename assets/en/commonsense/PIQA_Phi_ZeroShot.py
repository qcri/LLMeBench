from llmebench.datasets import PIQADataset
from llmebench.models import FastChatModel
from llmebench.tasks import PIQATask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "https://huggingface.co/microsoft/phi-1_5",
        "description": "Locally hosted Phi-1.5b model using FastChat.",
    }


def config():
    return {
        "dataset": PIQADataset,
        "task": PIQATask,
        "model": FastChatModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are an expert in Question Answering. You are concise as you only output the answer to the question without any illustrations or extra details.",
        },
        {
            "role": "user",
            "content": f'Question: {input_sample["goal"]},\nA. {input_sample["sol1"]}\nB. {input_sample["sol2"]} \nAnswer: ',
        },
    ]


def post_process(response):
    response = response["choices"][0]["message"]["content"]
    response = response.replace('"', "")
    response = response.strip()
    return response
