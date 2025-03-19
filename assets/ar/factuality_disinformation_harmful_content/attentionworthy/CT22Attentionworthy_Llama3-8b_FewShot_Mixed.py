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


def few_shot_prompt(input_sample, examples):
    base_prompt = (
        'قم بتصنيف "التغريدة" التالية إلى واحدة من الفئات التالية: '
        "yes_discusses_action_taken، harmful، yes_discusses_cure، yes_asks_question، no_not_interesting، yes_other، yes_blame_authorities، "
        "yes_contains_advice، yes_calls_for_action. قدم التصنيف فقط.\n\n"
        "إليك بعض الأمثلة:\n\n"
    )
    for index, example in enumerate(examples):
        base_prompt += (
            f"مثال {index + 1}:\n"
            f"التغريدة: '{example['input']}'\n"
            f"التصنيف: {example['label']}\n\n"
        )
    base_prompt += (
        f"الآن، قم بتقييم التغريدة الجديدة التالية:\n"
        f"التغريدة: '{input_sample}'\n"
        f"التصنيف: "
    )
    return base_prompt


def prompt(input_sample, examples):
    return [
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, examples),
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

    label_fixed = label.lower().strip()
    label_list = config()["model_args"]["class_labels"]

    label = label.replace("label:", "").strip()

    label = label.replace("<s>", "").replace("</s>", "")
    label_fixed = label.lower().strip()  # تحويل إلى أحرف صغيرة وإزالة الفراغات الزائدة

    label_fixed = label_fixed.replace("التصنيف:", "")

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
    elif label in label_list:
        label_fixed = label
    else:
        label_fixed = None

    return label_fixed
