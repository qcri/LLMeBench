import os

from arabic_llm_benchmark.datasets import StanceUnifiedFCDataset
from arabic_llm_benchmark.models import BLOOMPetalModel
from arabic_llm_benchmark.tasks import StanceUnifiedFCTask


def config():
    return {
        "dataset": StanceUnifiedFCDataset,
        "dataset_args": {},
        "task": StanceUnifiedFCTask,
        "task_args": {},
        "model": BLOOMPetalModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "class_labels": ["agree", "disagree", "discuss", "unrelated"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/factuality_stance_ramy/ramy_arabic_stance.jsonl"
        },
    }


def prompt(input_sample):
    article = input_sample["article"]
    article_arr = article.split()
    if len(article_arr) > 1000:
        article_str = " ".join(article_arr[:1000])
    else:
        article_str = article

    prompt_string = (
        f"Identify the stance of text with respect to the article as only agree, disagree, discuss or unrelated.\n"
        f'\ntext: {input_sample["claim"]}'
        f'\nclaim-text: {input_sample["claim-fact"]}'
        f"\narticle: {article_str}"
        f"\nstance: "
    )

    return {"prompt": prompt_string}


def post_process(response):
    label = response["outputs"].strip().lower()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")

    return label
