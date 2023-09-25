import re

from llmebench.datasets import WikiNewsSegmentationDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ArabicSegmentationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Accuracy": "0.252"},
    }


def config():
    return {
        "dataset": WikiNewsSegmentationDataset,
        "task": ArabicSegmentationTask,
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
            "content": f"A word can be composed of one root and one or multiple affixed, \
                    segment the following sentence into its morphological constituents:\n {input_sample}\
                    The input will be a list of words in the sentence. \
                    The output format should be a list of tuples, where each tuple consists of a word from the input text and its segmented form joined by a + sign.\
                    ",
        },
    ]


def post_process(response):
    results = []
    text = response["choices"][0]["message"]["content"]
    pattern = "\([\"']([^\"']+)[\"'], [\"']([^\"']+)[\"']\)"
    matches = re.finditer(pattern, text)
    for m in matches:
        results.append(m.group(2))

    text = " ".join(results)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text
