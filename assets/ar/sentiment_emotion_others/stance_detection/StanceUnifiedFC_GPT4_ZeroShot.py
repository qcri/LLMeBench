import os
import re

from llmebench.datasets import StanceUnifiedFCDataset
from llmebench.models import GPTChatCompletionModel
from llmebench.tasks import StanceUnifiedFCTask


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
            "data_path": "data/factuality_disinformation_harmful_content/factuality_stance_ramy/ramy_arabic_stance.jsonl"
        },
    }


def prompt(input_sample):
    claim = input_sample["claim"].strip()
    article = input_sample["article"].strip()

    # article = input_sample["article"]
    # article_arr = article.split()
    # if len(article_arr) > 2200:
    #     article_str = " ".join(article_arr[:2200])
    # else:
    # article_str = article

    # (agree, disagree, discuss, or unrelated)
    prompt_string = (
        f"Given a reference claim, and a news article, predict the stance of the article "
        f"towards the claim. Reply using one of these stances: 'agree' (if article agrees "
        f"with claim), 'disagree' (if article disagrees with claim), "
        f"'discuss' (if article discusses claim without specific stance), or 'unrelated' "
        f"(if article isn't discussing the claim's topic)"
        f"\n\n"
        f"reference claim: {claim}\n"
        f"news article: {article}\n"
        f"label: \n"
    )

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


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()
    label = label.replace("label:", "")
    label_fixed = label.lower()
    label_fixed = label_fixed.split()[0]
    label_fixed = label_fixed.replace(".", "")

    return label_fixed
