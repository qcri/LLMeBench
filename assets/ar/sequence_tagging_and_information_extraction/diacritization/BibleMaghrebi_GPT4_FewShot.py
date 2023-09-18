from llmebench.datasets import BibleMaghrebiDiacritizationDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ArabicDiacritizationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"WER": "0.994"},
    }


def config():
    return {
        "dataset": BibleMaghrebiDiacritizationDataset,
        "task": ArabicDiacritizationTask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
        "general_args": {
            "fewshot": {
                "train_split": ["morrocan_f05/dev", "tunisian_f05/dev"],
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
    base_prompt = f"Diacritize fully the following Arabic sentence including adding case endings:\n\
                     Make sure to put back non-Arabic tokens intact into the output sentence.\
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
    text = response["choices"][0]["message"]["content"]

    return text
