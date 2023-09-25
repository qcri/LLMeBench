from llmebench.datasets import WikiNewsDiacritizationDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ArabicDiacritizationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"WER": "0.420"},
    }


def config():
    return {
        "dataset": WikiNewsDiacritizationDataset,
        "task": ArabicDiacritizationTask,
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
            "content": f"Diacritize fully the following Arabic sentence including adding case endings:\n {input_sample}\n\
                     Make sure to put back non-Arabic tokens intact into the output sentence.\
                    ",
        },
    ]


def post_process(response):
    text = response["choices"][0]["message"]["content"]

    return text
