import random

from llmebench.datasets import TyDiQADataset
from llmebench.models import OpenAIModel
from llmebench.tasks import QATask

random.seed(3333)


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"F1": "0.739"},
    }


def config():
    return {
        "dataset": TyDiQADataset,
        "task": QATask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 30,
        },
        "general_args": {"test_split": "dev"},
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    output_prompt = base_prompt + "\n"
    for example in examples:
        context = example["input"]["context"]
        question = example["input"]["question"]
        answer = example["label"]
        output_prompt = (
            output_prompt
            + f"Context: {context}\nQuestion: {question}\nAnswer: {random.choice(answer)}\n"
        )

    input_context = input_sample["context"]
    input_question = input_sample["question"]
    output_prompt = (
        output_prompt
        + f"Context: {input_context}\n"
        + f"Question: {input_question}\n"
        + "Answer:"
    )
    return output_prompt


def prompt(input_sample, examples):
    base_prompt = f"Your task is to answer questions in Arabic based on a given context. \nNote: Your answers should be spans extracted from the given context without any illustrations.\nYou don't need to provide a complete answer."

    return [
        {
            "role": "system",
            "content": "Assistant is a large language model trained by OpenAI.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    return response["choices"][0]["message"]["content"]
