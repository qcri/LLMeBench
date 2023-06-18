import os

from arabic_llm_benchmark.datasets import StanceKhouja20Dataset
from arabic_llm_benchmark.models import BLOOMPetalModel
from arabic_llm_benchmark.tasks import StanceKhouja20Task


def config():
    return {
        "dataset": StanceKhouja20Dataset,
        "dataset_args": {},
        "task": StanceKhouja20Task,
        "task_args": {},
        "model": BLOOMPetalModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "class_labels": ["agree", "disagree"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/factuality_stance_khouja/stance/test.csv"
        },
    }


def prompt(input_sample):
    prompt = f'Can you check if first sentence agree or disagree with second sentence? Say only agree or disagree.\n\n first-sentence: {input_sample["sentence_1"]}\nsecond-sentence: {input_sample["sentence_2"]}\n label: \n'

    return {"prompt": prompt}


def post_process(response):
    label = response["outputs"].strip().lower()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")
    print(label)
    return label


# def post_process(response):
#     label = response["choices"][0]["text"].lower().replace(".", "")

#     return label
