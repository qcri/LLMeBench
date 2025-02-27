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
    base_prompt = f'قم بتصنيف "التغريدة" التالية إلى واحدة من الفئات التالية: يناقش_الإجراء_المتخذ، ضار، يناقش_العلاج، يطرح_سؤال، غير_مثير_للاهتمام، آخر، يلوم_السلطات، يحتوي_على_نصيحة، يدعو_لإجراء. قدم التصنيف فقط.\n\n'
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
        # إذا كان "المحتوى" في استجابة "الرسائل"
        label = response["output"].strip()
        label = label.replace("<s>", "")
        label = label.replace("</s>", "")
    else:
        print("استجابة .. " + str(response))
        label = ""

    label_fixed = label.lower().strip()  # تحويل إلى أحرف صغيرة وإزالة الفراغات الزائدة

    label_fixed = label_fixed.replace("التصنيف:", "")
    if label_fixed.startswith("لا"):
        label_fixed = "no_not_interesting"
    elif "يناقش_العلاج" in label_fixed:
        label_fixed = "yes_discusses_cure"
    elif "ضار" in label_fixed:
        label_fixed = "harmful"
    elif "يلوم_السلطات" in label_fixed:
        label_fixed = "yes_blame_authoritie"
    elif "يدعو_لإجراء" in label_fixed:
        label_fixed = "yes_calls_for_action"
    elif "يناقش_الإجراء_المتخذ" in label_fixed:
        label_fixed = "yes_discusses_action_taken"
    elif "ضار" in label_fixed:
        label_fixed = "harmful"
    elif "علاج" in label_fixed:
        label_fixed = "yes_discusses_cure"
    elif "يطرح سؤال" in label_fixed:
        label_fixed = "yes_asks_question"
    elif "غير_مثير_للاهتمام" in label_fixed:
        label_fixed = "no_not_interesting"
    elif "آخر" in label_fixed:
        label_fixed = "yes_other"
    elif "السلطات" in label_fixed:
        label_fixed = "yes_blame_authorities"
    elif "نصيحة" in label_fixed:
        label_fixed = "yes_contains_advice"
    elif "يدعو لإجراء" in label_fixed:
        label_fixed = "yes_calls_for_action"
    else:
        label_fixed = None

    return label_fixed
