import os

from arabic_llm_benchmark.datasets import ArabicDiacritizationDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import ArabicDiacritizationTask


def config():
    sets = [
        ('mor', 'morrocan_f05.test.src-trg.txt', 'morrocan_f05.dev.src-trg.txt'),
        ('tun', 'tunisian_f05.test.src-trg.txt', 'tunisian_f05.dev.src-trg.txt'),
        ('msa', 'WikiNewsTruth.txt', 'WikiNewsTruthDev.txt')
    ]
    configs = []
    for name, testset, devset in sets:
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
                    "data_path": "data/sequence_tagging_ner_pos_etc/diacritization/"+testset,
                "fewshot": { 
                            "train_data_path":  "data/sequence_tagging_ner_pos_etc/diacritization/" + devset
                        }
                },
            }})
    return configs

def few_shot_prompt(input_sample, base_prompt, examples):
    output_prompt = base_prompt + "\n"
    for example in examples:
        tokens = example["input"]
        label = example["label"]
        output_prompt = (
            output_prompt + f"Sentence: {tokens}\nLabels: {label}\n"
        )
    output_prompt = output_prompt + f"Sentence: {input_sample}\n" + "Labels:"
    return output_prompt


def prompt(input_sample, examples):
    base_prompt = f"Diacritize fully the following Arabic sentence including adding case endings:\n\
                     Make sure to put back non-Arabic tokens intact into the output sentence.\
                    "
    return [
                {"role":"system","content": "You are a linguist that helps in annotating data."},
                {
                    "role":"user",
                    "content": few_shot_prompt(input_sample, base_prompt, examples),
                } 
            ]

def post_process(response):
   
    text = response["choices"][0]["message"]["content"]
    
    return text

