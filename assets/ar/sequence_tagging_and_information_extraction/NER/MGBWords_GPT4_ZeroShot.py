import re

from llmebench.datasets import MGBWordsDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import NERTask


def config():
    return {
        "dataset": MGBWordsDataset,
        "dataset_args": {},
        "task": NERTask,
        "task_args": {},
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
                "O",
            ],
            "max_tries": 150,
        },
        "general_args": {
            "data_path": "data/sequence_tagging_ner_pos_etc/NER/mgb/MGB-words.txt"
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "Assistant is a large language model trained by OpenAI.",
        },
        {
            "role": "user",
            "content": f"Task Description: You are working as a named entity recognition expert and your task is to label a given arabic text with named entity labels. Your task is to identify and label any named entities present in the text. The named entity labels that you will be using are PER (person), LOC (location), ORG (organization) and MISC (miscellaneous). You may encounter multi-word entities, so make sure to label each word of the entity with the appropriate prefix ('B' for first word entity, 'I' for any non-initial word entity). For words which are not part of any named entity, you should return 'O'.\nNote: Your output format should be a list of tuples, where each tuple consists of a word from the input text and its corresponding named entity label.\nInput:{input_sample.split()}",
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
