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
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/factuality_stance_ramy/ramy_arabic_stance.jsonl",
            "fewshot": {
                "train_data_path": "data/factuality_disinformation_harmful_content/factuality_stance_khouja/stance/train.csv"
            },
        },
    }


def prompt(input_sample, examples):
    prompt_string = (
        f"Given a reference claim, and a news article, predict the stance of the article "
        f"towards the claim. Reply using one of these stances: 'agree' (if article agrees "
        f"with claim), 'disagree' (if article disagrees with claim), "
        f"'discuss' (if article discusses claim without specific stance), or 'unrelated' "
        f"(if article isn't discussing the claim's topic)"
        f"\n\n"
    )

    prompt_string = few_shot_prompt(input_sample, prompt_string, examples)

    return [
        {
            "role": "system",
            "content": "You are a fact checking expert.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt
    for example in examples:
        ref_s = example["input"].split("\t")[0]
        claim = example["input"].split("\t")[1]
        label = "unrelated" if example["label"] == "other" else example["label"]

        out_prompt = (
            out_prompt
            + "reference claim: "
            + ref_s
            + "\nnews article: "
            + claim
            + "\nlabel: "
            + label
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the label blank

    claim, article = input_sample.split("article: ")
    claim = claim.replace("claim:", " ").strip()
    article = article.strip()

    out_prompt = (
        out_prompt
        + f"reference claim: {claim}\n"
        + f"news article: {article}\nlabel: \n"
    )

    # print("=========== FS Prompt =============\n")
    # print(out_prompt)

    return out_prompt


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()
    label = label.replace("label:", "")
    label_fixed = label.lower()
    label_fixed = label_fixed.split()[0]
    label_fixed = label_fixed.replace(".", "")

    return label_fixed
