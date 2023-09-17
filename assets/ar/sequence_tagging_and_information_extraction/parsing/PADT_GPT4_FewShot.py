import re

from llmebench.datasets import PADTDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ArabicParsingTask


def config():
    return {
        "dataset": PADTDataset,
        "dataset_args": {},
        "task": ArabicParsingTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sequence_tagging_ner_pos_etc/parsing/arabic_PADT_test_gs.conll",
            "fewshot": {
                "train_data_path": "data/sequence_tagging_ner_pos_etc/parsing/arabic_PADT_train.conll"
            },
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
