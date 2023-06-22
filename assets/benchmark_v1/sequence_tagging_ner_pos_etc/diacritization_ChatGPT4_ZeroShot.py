import os

from arabic_llm_benchmark.datasets import ArabicDiacritizationDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import ArabicDiacritizationTask


def config():
    sets = [
        ('mor','morrocan_f05.test.src-tgt.txt'),
        ('tun','tunisian_f05.test.src-tgt.txt'),
        ('msa','WikiNewsTruth.txt')
    ]
    configs = []
    for name, testset in sets:
        configs.append({
            "name": name,
            "config": {
                "dataset": ArabicDiacritizationDataset,
                "dataset_args": {},
                "task": ArabicDiacritizationTask,
                "task_args": {},
                "model": GPTChatCompletionModel,
                "model_args": {
                    "api_type": "azure",
                    "api_version": "2023-03-15-preview",
                    "api_base": os.environ["AZURE_API_URL"],
                    "api_key": os.environ["AZURE_API_KEY"],
                    "engine_name": os.environ["ENGINE_NAME"],
                    "max_tries": 3,
                },
                "general_args": {
                    "data_path": "data/sequence_tagging_ner_pos_etc/diacritization/"+testset
                },
            }})
    return configs


def prompt(input_sample):
    return [
                {"role":"system","content": "You are a linguist that helps in annotating data."},
                {
                    "role":"user",
                    "content":f"Diacritize fully the following Arabic sentence including adding case endings:\n {input_sample}\n\
                     Make sure to put back non-Arabic tokens intact into the output sentence.\
                    ",
                } 
            ]

def post_process(response):
   
    text = response["choices"][0]["message"]["content"]
    
    return text

