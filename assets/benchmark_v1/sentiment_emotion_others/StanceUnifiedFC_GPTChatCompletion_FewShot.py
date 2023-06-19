import os
import random
import re

from arabic_llm_benchmark.datasets import StanceUnifiedFCDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import StanceUnifiedFCTask


random.seed(1333)


def config():
    return {
        "dataset": StanceUnifiedFCDataset,
        "dataset_args": {},
        "task": StanceUnifiedFCTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["agree", "disagree", "unrelated"],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/factuality_stance_ramy/ramy_arabic_stance.jsonl",
            "fewshot": {
                "train_data_path": "data/factuality_disinformation_harmful_content/attentionworthy/CT22_arabic_1D_attentionworthy_train.tsv",
            },
        },
    }

def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    for example in examples:
        out_prompt = (
            out_prompt
            + f"claim: {example['claim']}\n"
            + f'\nclaim\'s text: {example["claim-fact"]}'
            + f"news Article: {example['article']}\n\n"
            + f"stance: example['stance']\n"
            + f"\n\n"
        )

    input_example = (
        f"claim: {input_sample['claim']}\n"
        f'\nclaim\'s text: {input_sample["claim-fact"]}'
        f"news Article: {input_sample['article']}\n"
    )
            
    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + input_example + "\nstance: \n"

    return out_prompt


def prompt(input_sample, examples):
    article = input_sample["article"]
    # article_arr = article.split()
    # if len(article_arr) > 2200:
    #     article_str = " ".join(article_arr[:2200])
    # else:
    article_str = article

    base_prompt = "Based on your analysis, determine the stance of the news article towards the claim. The possible stances could be 'agree', 'disagree', or 'unrelated'."

    return [
        {
            "role": "system",
            "content": "You are a fact checking expert. Your task is to analyze the claim, text of the claim and the associated news article.",
        },
        {
            "role": "user",
            # "content": prompt_string,
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()
    label = label.replace("stance:", "")
    label_fixed = label.lower()
    label_fixed = label_fixed.split()[0]
    label_fixed = label_fixed.replace(".", "")

    return label_fixed
