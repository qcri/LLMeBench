import re

from llmebench.datasets import PADTDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ArabicParsingTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"UAS": "0.551"},
    }


def config():
    return {
        "dataset": PADTDataset,
        "task": ArabicParsingTask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    output_prompt = base_prompt + "\n"
    for example in examples:
        tokens = example["input"]
        label = example["label"]
        output_prompt = output_prompt + f"Sentence: {tokens}\nLabels: {label}\n"
    output_prompt = output_prompt + f"Sentence: {input_sample}\n" + "Labels:"
    return output_prompt


def prompt(input_sample, examples):
    base_prompt = f"Given the following features (in order: ID, Form, Lemma, CPostTag, POSTag, Features),\n\
                predict the Head of each token in the following sentence, which is either a value of a related ID or 0.\n\
                A value of zero means the token attaches to the virtual root node:\n"

    return [
        {
            "role": "system",
            "content": "You are a linguist that helps in annotating data.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    output = response["choices"][0]["message"]["content"]
    pattern = "[\"']([^\"']+)[\"']: [\"']([^\"']+)[\"']"
    matches = re.finditer(pattern, output)
    results = {}
    for m in matches:
        results[m.group(1)] = m.group(2)
    return results
