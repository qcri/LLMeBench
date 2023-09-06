import os

from llmebench.datasets import Khouja20StanceDataset
from llmebench.models import GPTChatCompletionModel
from llmebench.tasks import StanceTask


def config():
    return {
        "dataset": Khouja20StanceDataset,
        "dataset_args": {},
        "task": StanceTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["agree", "disagree"],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/factuality_stance_khouja/stance/test.csv",
            "fewshot": {
                "train_data_path": "data/factuality_disinformation_harmful_content/factuality_stance_khouja/stance/train.csv"
            },
        },
    }


def prompt(input_sample, examples):
    prompt_string = "Given a reference sentence and a claim, predict whether the claim agrees or disagrees with the reference sentence. Reply only using 'agree', 'disagree', or use 'other' if the sentence and claim are unrelated.\n\n"

    return [
        {
            "role": "system",
            "content": "You are a fact checking expert.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, prompt_string, examples),
        },
    ]


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt
    for example in examples:
        ref_s = example["input"]["sentence_1"]
        claim = example["input"]["sentence_2"]

        out_prompt = (
            out_prompt
            + "reference sentence: "
            + ref_s
            + "\nclaim: "
            + claim
            + "\nlabel: "
            + example["label"]
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the label blank

    ref_s = input_sample["sentence_1"]
    claim = input_sample["sentence_2"]

    out_prompt = (
        out_prompt
        + "reference sentence: "
        + ref_s
        + "\nclaim: "
        + claim
        + "\nlabel: \n"
    )

    # print("=========== FS Prompt =============\n")
    # print(out_prompt)

    return out_prompt


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()
    label = label.replace("label:", "").strip()

    label_fixed = None

    if "unrelated" in label or "other" in label:
        label_fixed = "other"
    elif "disagree" in label:
        label_fixed = "disagree"
    elif label == "agree":
        label_fixed = "agree"

    return label_fixed
