import re

from llmebench.datasets import AqmarDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import NERTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Macro F1": "0.390"},
    }


def config():
    return {
        "dataset": AqmarDataset,
        "task": NERTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": [
                "B-PERS",
                "I-PERS",
                "B-LOC",
                "I-LOC",
                "B-ORG",
                "I-ORG",
                "B-MISC",
                "I-MISC",
            ],
            "max_tries": 50,
        },
        "general_args": {"fewshot": {"train_split": "dev"}},
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    output_prompt = base_prompt + "\n"
    for example in examples:
        tokens = example["input"].split()
        label = example["label"].split()
        few_shot_sample = list(zip(tokens, label))
        output_prompt = (
            output_prompt + f"Sentence: {tokens}\nLabels: {few_shot_sample}\n"
        )
    output_prompt = output_prompt + f"Sentence: {input_sample}\n" + "Labels:"
    return output_prompt


def prompt(input_sample, examples):
    base_prompt = f"Task Description: You are working as a named entity recognition expert and your task is to label a given arabic text with named entity labels. Your task is to identify and label any named entities present in the text. The named entity labels that you will be using are PER (person), LOC (location), ORG (organization) and MISC (miscellaneous). You may encounter multi-word entities, so make sure to label each word of the entity with the appropriate prefix ('B' for first word entity, 'I' for any non-initial word entity). For words which are not part of any named entity, you should return 'O'.\nNote: Your output format should be a list of tuples, where each tuple consists of a word from the input text and its corresponding named entity label."
    return [
        {
            "role": "system",
            "content": "Assistant is a large language model trained by OpenAI.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    response = response["choices"][0]["message"]["content"]
    possible_tags = [
        "B-PER",
        "I-PER",
        "B-LOC",
        "I-LOC",
        "B-ORG",
        "I-ORG",
        "O",
        "B-MISC",
        "I-MISC",
    ]
    mapping = {
        "PER-B": "B-PER",
        "PER-I": "I-PER",
        "ORG-B": "B-ORG",
        "ORG-I": "I-ORG",
        "LOC-B": "B-LOC",
        "LOC-I": "I-LOC",
        "MISC-B": "B-MISC",
        "MISC-I": "I-MISC",
    }

    matches = re.findall(r"\((.*?)\)", response)
    if matches:
        cleaned_response = []
        for match in matches:
            elements = match.split(",")
            try:
                cleaned_response.append(elements[1])
            except:
                cleaned_response.append("O")

        cleaned_response = [
            sample.replace("'", "").strip() for sample in cleaned_response
        ]
        final_cleaned_response = []
        for elem in cleaned_response:
            if elem in possible_tags:
                final_cleaned_response.append(elem)
            elif elem in mapping:
                final_cleaned_response.append(mapping[elem])
            else:
                final_cleaned_response.append("O")
    else:
        final_cleaned_response = None
    return final_cleaned_response
