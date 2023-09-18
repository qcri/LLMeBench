import re

from llmebench.datasets import PADTDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ArabicParsingTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"UAS": "0.504"},
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


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are a linguist that helps in annotating data.",
        },
        {
            "role": "user",
            "content": f"Given the following features (in order: ID, Form, Lemma, CPostTag, POSTag, Features), predict the Head of each token in the following sentence, which is either a value of a related ID or 0. A value of zero means the token attaches to the virtual root node: {input_sample}",
        },
    ]


def post_process(response):
    output = response["choices"][0]["message"]["content"]
    if len(output):
        output = output.strip().split("\n")
        results = {}
        for o in output:
            src, tgt = (re.sub(r"[^0-9]+", "\t", o)).split("\t")
            results[src] = tgt
        output = results
    else:
        output = None
    return output
