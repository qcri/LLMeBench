from llmebench.datasets import WikiNewsLemmatizationDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import LemmatizationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-35-turbo (version 0301)",
        "description": "GPT35 model hosted on Azure, using the Completion API. API version '2023-03-15-preview'.",
        "scores": {"Accuracy": "0.471"},
    }


def config():
    return {
        "dataset": WikiNewsLemmatizationDataset,
        "dataset_args": {},
        "task": LemmatizationTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"for every word in the following Arabic sentence, write only the lemma without diacritics separated by a single space without explanation:\n {input_sample}",
            }
        ],
    }


def post_process(response):
    out = response["choices"][0]["text"].strip()

    # TODO: fix hack to handle prediction failure
    return (None, out)
