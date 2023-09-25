from llmebench.datasets import UnifiedFCStanceDataset
from llmebench.models import PetalsModel
from llmebench.tasks import StanceTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"Macro-F1": "0.235"},
    }


def config():
    return {
        "dataset": UnifiedFCStanceDataset,
        "task": StanceTask,
        "model": PetalsModel,
        "model_args": {
            "class_labels": ["agree", "disagree", "discuss", "unrelated"],
            "max_tries": 3,
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
