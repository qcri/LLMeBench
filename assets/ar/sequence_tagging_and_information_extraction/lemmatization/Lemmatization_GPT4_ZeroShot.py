import os

from llmebench.datasets import LemmatizationDataset
from llmebench.models import GPTChatCompletionModel
from llmebench.tasks import LemmatizationTask


def config():
    return {
        "dataset": LemmatizationDataset,
        "dataset_args": {},
        "task": LemmatizationTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sequence_tagging_ner_pos_etc/lemmatization/WikiNews-26-06-2015-RefLemma.txt"
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
