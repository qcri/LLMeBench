import os

from arabic_llm_benchmark.datasets import MGBWordsDataset
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel
from arabic_llm_benchmark.tasks import NERTask


def config():
    return {
        "dataset": MGBWordsDataset,
        "dataset_args": {},
        "task": NERTask,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "gpt",
            "class_labels": [[]],
            "max_tries": 3,
        },
        "general_args": {"data_path": "data/speech/NER/MGB-words.txt"},
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Task Description: You are working as a named entity recognition expert and your task is to label a given arabic text with named entity labels. Your task is to identify and label any named entities present in the text without any explanation. The named entity labels that you will be using are PER (person), LOC (location), and ORG (organization). You may encounter multi-word entities, so make sure to label each word of the entity with the appropriate prefix ('B' for first word entity, 'I' for any non-initial word entity). For words which are not part of any named entity, you should return 'O'. Note: Your output format should be a list of tuples, where each tuple consists of a word from the input text and its corresponding named entity label. Input: {input_sample}",
            }
        ],
    }


def parse_ner_labels(chat_out):
    chat_out = chat_out.replace("),", "\t")
    fields = chat_out[1 : len(chat_out) - 2].split("\t")  # ignore []
    labels = []
    for f in fields:
        fields2 = f.split(",")
        word = fields2[0]
        label = fields2[1]
        # print(f)
        # print(word, label, "\n")
        label = label.strip().replace("'", "")
        # print(label)
        if label == "PER":
            label = "B-PER"
        elif label == "LOC":
            label = "B-LOC"
        elif label == "ORG":
            label = "B-ORG"
        elif label == "OTH":
            label = "B-OTH"

        labels.append(label)
    # print("\n", labels)
    return labels


def post_process(response):
    if response is None:
        return []
    return parse_ner_labels(response["choices"][0]["text"])
