from llmebench.datasets import AdultDataset
from llmebench.models import AzureModel
from llmebench.tasks import AdultTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "LLama 3 8b",
        "description": "Deployed on Azure.",
        "scores": {"Macro-F1": "0.3731"},
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
    base_prompt = "هل تحتوي التغريدة التالية على لغة للبالغين؟ اجب 'adult' أو 'not adult' بدون أي تفسير.\n\n"
    for index, example in enumerate(examples):
        label = "adult" if example["label"] == "ADULT" else "not adult"
        base_prompt += (
            f"مثال {index + 1}:\n"
            f"جملة: '{example['input']}'\n"
            f"التصنيف: {label}\n\n"
        )
    base_prompt += f"صنف الآن الجملة الجديدة:\nجملة: '{input_sample}'\n" f"التصنيف: "
    return base_prompt


def prompt(input_sample, examples):
    return [
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, examples),
        }
    ]


def post_process(response):

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
