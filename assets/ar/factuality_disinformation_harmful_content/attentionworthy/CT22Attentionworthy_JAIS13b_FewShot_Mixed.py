from llmebench.datasets import CT22AttentionworthyDataset
from llmebench.models import FastChatModel
from llmebench.tasks import AttentionworthyTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "LLama3-8b",
        "description": "Deployed on Azure.",
        "scores": {"Weighted-F1": "0.412"},
    }


def config():
    return {
        "dataset": CT22AttentionworthyDataset,
        "task": AttentionworthyTask,
        "model": FastChatModel,
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
            "max_tries": 3,
        },
        "general_args": {"test_split": "ar", "fewshot": {"train_split": "ar"}},
    }


# def few_shot_prompt(input_sample, examples):
#     base_prompt = (
#         'قم بتصنيف "التغريدة" التالية إلى واحدة من الفئات التالية: '
#         'yes_discusses_action_taken، harmful، yes_discusses_cure، yes_asks_question، no_not_interesting، yes_other، yes_blame_authorities، '
#         'yes_contains_advice، yes_calls_for_action. قدم التصنيف فقط.\n\n'
#         "إليك بعض الأمثلة:\n\n"
#     )
#     for index, example in enumerate(examples):
#         base_prompt += (
#             f"مثال {index + 1}:\n"
#             f"التغريدة: '{example['input']}'\n"
#             f"التصنيف: {example['label']}\n\n"
#         )
#     base_prompt += (
#         f"الآن، قم بتقييم التغريدة الجديدة التالية:\n"
#         f"التغريدة: '{input_sample}'\n"
#         f"التصنيف: "
#     )
#     return base_prompt


# def prompt(input_sample, examples):
#     return [
#         {
#             "role": "user",
#             "content": few_shot_prompt(input_sample, examples),
#         },
#     ]
def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    for example in examples:
        out_prompt = (
            out_prompt
            + "التغريدة: "
            + example["input"]
            + "\التصنيف: "
            + example["label"]
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "التغريدة: " + input_sample + "\التصنيف: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = (
        'قم بتصنيف "التغريدة" التالية إلى واحدة من الفئات التالية: '
        "yes_discusses_action_taken، harmful، yes_discusses_cure، yes_asks_question، no_not_interesting، yes_other، yes_blame_authorities، "
        "yes_contains_advice، yes_calls_for_action. قدم التصنيف فقط.\n\n"
        "إليك بعض الأمثلة:\n\n"
    )
    return [
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


import re


def post_process(response):
    print(response)
    label = response["choices"][0]["message"]["content"]
    label_list = config()["model_args"]["class_labels"]

    label = label.replace("label:", "").strip()

    label = label.replace("<s>", "").replace("</s>", "")
    label_fixed = label.lower().strip()  # تحويل إلى أحرف صغيرة وإزالة الفراغات الزائدة

    label_fixed = label_fixed.replace("التصنيف:", "")

    if label.startswith("no"):
        label_fixed = "no_not_interesting"
    elif label == "yes_discusses_covid-19_vaccine_side_effects":
        label_fixed = "yes_discusses_cure"
    elif label == "yes_harmful":
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
    elif label in label_list:
        label_fixed = label
    else:
        label_fixed = None

    return label_fixed
