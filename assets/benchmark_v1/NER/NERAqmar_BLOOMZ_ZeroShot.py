import os

from arabic_llm_benchmark.datasets import AqmarDataset
from arabic_llm_benchmark.models import BLOOMPetalModel
from arabic_llm_benchmark.tasks import NERTask


def config():
    return {
        "dataset": AqmarDataset,
        "dataset_args": {},
        "task": NERTask,
        "task_args": {},
        "model": BLOOMPetalModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "max_tries": 100,
        },
        
        "general_args": {
            "data_path": {
                "split": "test",
                "path": "data/sequence_tagging_ner_pos_etc/NER/aqmar/AQMAR_Arabic_NER_corpus-1.0",
            },
        },
    }


def prompt(input_sample):
    return {
        "prompt": f"Task Description: You are working as a named entity recognition expert and your task is to label a given arabic text with named entity labels. Your task is to identify and label any named entities present in the text. The named entity labels that you will be using are PER (person), LOC (location), ORG (organization) and MISC (miscellaneous). You may encounter multi-word entities, so make sure to label each word of the entity with the appropriate prefix ('B' for first word entity, 'I' for any non-initial word entity). For words which are not part of any named entity, you should return 'O'.\nNote: Your output format should be a list of tuples, where each tuple consists of a word from the input text and its corresponding named entity label.\nInput:{input_sample.split()}"
    }


def post_process(response):
    return response["outputs"]