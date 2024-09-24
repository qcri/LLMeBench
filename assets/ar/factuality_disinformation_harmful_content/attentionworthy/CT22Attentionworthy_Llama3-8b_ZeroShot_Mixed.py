from llmebench.datasets import CT22AttentionworthyDataset
from llmebench.models import AzureModel
from llmebench.tasks import AttentionworthyTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "LLama3-8b",
        "description": "Deployed on Azure.",
        "scores": {"Weighted-F1": "0.257"},
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
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    base_prompt = (
        'قم بتصنيف "التغريدة" التالية إلى واحدة من الفئات التالية: '
        "yes_discusses_action_taken، harmful، yes_discusses_cure، yes_asks_question، no_not_interesting، yes_other، yes_blame_authorities، "
        "yes_contains_advice، yes_calls_for_action. قدم التصنيف فقط.\n\n"
        f"التغريدة: '{input_sample}'\n"
        "التصنيف: "
    )
    return [
        {
            "role": "user",
            "content": base_prompt,
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
