from llmebench.datasets import WikiNewsLemmatizationDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import LemmatizationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Accuracy": "0.397"},
    }


def config():
    return {
        "dataset": WikiNewsLemmatizationDataset,
        "task": LemmatizationTask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are a language expert, you can identify the lemma of any word within a sentence.",
        },
        {
            "role": "user",
            "content": f"for every word in the following Arabic word, write only the lemma without diacritics separated by a single space without explanation:\n {input_sample}",
        },
    ]


def post_process(response):
    x = response["choices"][0]["message"]["content"]
    if (
        x.startswith("Please provide the Arabic sentence")
        or x.startswith("It seems")
        or "is not" in x
    ):
        out = None
    else:
        # TODO: fix hack to handle prediction failure
        out = (None, x)

    return out
