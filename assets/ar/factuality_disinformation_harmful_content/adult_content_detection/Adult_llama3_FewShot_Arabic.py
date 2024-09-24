from llmebench.datasets import AdultDataset
from llmebench.models import AzureModel
from llmebench.tasks import AdultTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "LLama 3 8b",
        "description": "Deployed on Azure.",
    }


def config():
    return {
        "dataset": AdultDataset,
        "task": AdultTask,
        "model": AzureModel,
        "model_args": {
            "class_labels": ["ADULT", "NOT_ADULT"],
            "max_tries": 3,
        },
    }


def few_shot_prompt(input_sample, examples):
    base_prompt = (
        "صنف التغريدة العربية التالية على أنها تحتوي على لغة للبالغين (اللغة المستخدمة في إعلانات البالغين والإعلانات الإباحية) أو لا تحتوي على لغة للبالغين بدون توضيح. "
        "في حالة لغة البالغين، اكتب فقط 'للبالغين' بدون تفسير، وفي حالة عدم احتوائها على لغة البالغين، اكتب فقط 'ليست للبالغين' بدون تفسير:\n\n"
    )
    for index, example in enumerate(examples):
        label = "للبالغين" if example["label"] == "ADULT" else "ليست للبالغين"
        base_prompt += (
            f"مثال {index + 1}:\n"
            f"التغريدة: '{example['input']}'\n"
            f"التصنيف: {label}\n\n"
        )
    base_prompt += (
        f"صنف الآن التغريدة الجديدة:\nالتغريدة: '{input_sample}'\n" f"التصنيف: "
    )
    return base_prompt


def prompt(input_sample, examples):
    return [
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, examples),
        }
    ]


def post_process(response):
    # if not response or 'error' in response or 'output' not in response:
    # print("Error or missing output in response:", response)
    # return "NOT_ADULT"  # Safely default to NOT_ADULT when unsure

    label = response["output"].strip().lower()
    label = label.replace("التصنيف:", "").strip()
    label = label.replace("label:", "").strip()

    label = label.replace("<s>", "").replace("</s>", "")
    label = label.lower()

    if (
        "ليس" in label
        or "ليست" in label
        or "not" in label
        or "no" in label
        or "غير" in label
        or "لا" in label
    ):
        return "NOT_ADULT"
    elif "للبالغين" in label or "نعم" in label or "adult" in label or "بالغين" in label:
        return "ADULT"
    else:
        return None
