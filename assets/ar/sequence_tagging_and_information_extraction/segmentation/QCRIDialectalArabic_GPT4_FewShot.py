import re

from llmebench.datasets import QCRIDialectalArabicSegmentationDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ArabicSegmentationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Accuracy (Avg)": "0.850"},
    }


def config():
    return {
        "dataset": QCRIDialectalArabicSegmentationDataset,
        "task": ArabicSegmentationTask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
        "general_args": {
            "fewshot": {
                "train_split": [
                    "glf.data_5/dev",
                    "lev.data_5/dev",
                    "egy.data_5/dev",
                    "mgr.data_5/dev",
                ],
            }
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
    base_prompt = f"A word can be composed of one root and one or multiple affixed, \
                    segment the following sentence into its morphological constituents.\
                    The input will be a list of words in the sentence. \
                    The output format should be a list where each word from the input text is segmented and joined by a + sign.\
                    "

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
    results = []
    text = response["choices"][0]["message"]["content"]
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text
