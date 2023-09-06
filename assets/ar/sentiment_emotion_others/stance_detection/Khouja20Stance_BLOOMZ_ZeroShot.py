import os

from llmebench.datasets import Khouja20StanceDataset
from llmebench.models import PetalsModel
from llmebench.tasks import StanceTask


def config():
    return {
        "dataset": Khouja20StanceDataset,
        "dataset_args": {},
        "task": StanceTask,
        "task_args": {},
        "model": PetalsModel,
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

    return label
