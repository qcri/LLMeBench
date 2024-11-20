from llmebench.datasets import CT22AttentionworthyDataset
from llmebench.models import AzureModel
from llmebench.tasks import AttentionworthyTask


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "Llama-3.1-8B-Instruct",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": CT22AttentionworthyDataset,
        "task": AttentionworthyTask,
        "model": AzureModel,
        "model_args": {
            "class_labels": [
                "yes_discusses_action_taken",
                "harmful",
                "yes_discusses_cure",
                "yes_asks_question",
                "no_not_interesting",
                "yes_other",
                "yes_blame_authorities",
                "yes_contains_advice",
                "yes_calls_for_action",
            ],
            "max_tries": 100,
        },
        "general_args": {"test_split": "ar", "fewshot": {"train_split": "ar"}},
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    for example in examples:
        out_prompt = (
            out_prompt
            + "tweet: "
            + example["input"]
            + "\nlabel: "
            + example["label"]
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "tweet: " + input_sample + "\nlabel: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = f'Annotate the following "tweet" into one of the following categories: yes_discusses_action_taken, harmful, yes_discusses_cure, yes_asks_question, no_not_interesting, yes_other, yes_blame_authorities, yes_contains_advice, yes_calls_for_action. Provide only label.\n\n'
    return [
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


import re


def post_process(response):
    print(response)
    if "output" in response:
        # if "content" in response["messages"]:
        label = response["output"].strip()
        label = label.replace("<s>", "")
        label = label.replace("</s>", "")
    else:
        print("Response .. " + str(response))
        label = ""

    label_fixed = (
        label.lower().strip()
    )  # Convert to lowercase and strip leading/trailing whitespace

    if label.startswith("no"):
        label_fixed = "no_not_interesting"
    elif "yes_discusses_covid-19_vaccine_side_effects" in label:
        label_fixed = "yes_discusses_cure"
    elif "yes_harmful" in label:
        label_fixed = "harmful"
    elif label.startswith("yes"):
        label_fixed = label
    elif "yes_blame_authoritie" in label:
        label_fixed = "yes_blame_authoritie"
    elif "yes_discusses_action_taken" in label:
        label_fixed = "yes_discusses_action_taken"
    elif "harmful" in label:
        label_fixed = "harmful"
    elif "yes_discusses_cure" in label:
        label_fixed = "yes_discusses_cure"
    elif "yes_asks_question" in label:
        label_fixed = "yes_asks_question"
    elif "no_not_interesting" in label:
        label_fixed = "no_not_interesting"
    elif "yes_other" in label:
        label_fixed = "yes_other"
    elif "yes_blame_authorities" in label:
        label_fixed = "yes_blame_authorities"
    elif "yes_contains_advice" in label:
        label_fixed = "yes_contains_advice"
    elif "yes_calls_for_action" in label:
        label_fixed = "yes_calls_for_action"
    else:
        label_fixed = None

    return label_fixed
