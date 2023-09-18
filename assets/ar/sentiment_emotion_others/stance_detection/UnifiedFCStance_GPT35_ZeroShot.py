from llmebench.datasets import UnifiedFCStanceDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import StanceTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-35-turbo (version 0301)",
        "description": "GPT35 model hosted on Azure, using the Completion API. API version '2023-03-15-preview'.",
        "scores": {"Macro-F1": "0.232"},
    }


def config():
    return {
        "dataset": UnifiedFCStanceDataset,
        "dataset_args": {},
        "task": StanceTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "class_labels": ["agree", "disagree", "unrelated"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    article = input_sample["article"]
    article_arr = article.split()
    if len(article_arr) > 2200:
        article_str = " ".join(article_arr[:2200])
    else:
        article_str = article

    prompt_string = (
        f"Identify the stance of text with respect to the article as only agree, disagree, discuss or unrelated.\n"
        f'\ntext: {input_sample["claim"]}'
        f'\nclaim-text: {input_sample["claim-fact"]}'
        f"\narticle: {article_str}"
        f"\nstance: \n"
    )

    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": prompt_string,
            }
        ],
    }


def post_process(response):
    label = response["choices"][0]["text"].lower().replace(".", "")
    label_fixed = label.lower()
    label_fixed = label_fixed.split()[0]
    label_fixed = label_fixed.replace(".", "")

    return label_fixed
