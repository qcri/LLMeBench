from llmebench.datasets import AdultDataset
from llmebench.models import FastChatModel
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
        "model": FastChatModel,
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
            f"التغريدة: '{example['input']}'\n"
            f"التصنيف: {label}\n\n"
        )
    base_prompt += f"صنف الآن التغريدة الجديدة:\nجملة: '{input_sample}'\n" f"التصنيف: "
    return base_prompt


def prompt(input_sample, examples):
    return [
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, examples),
        }
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]
    label = label.replace("التصنيف:", "").strip()
    label = label.replace("label:", "").strip()

    label = label.replace("<s>", "").replace("</s>", "")
    label = label.lower()

    if (
        "cannot classify" in label
        or "cannot complete" in label
        or "لا يمكنني" in label
        or "cannot fulfill" in label
    ):
        return None
    elif (
        "غير مناسب للبالغين" in label
        or "غير مناسبة للبالغين" in label
        or "مناسب للجمهور العام" in label
    ):
        return "NOT_ADULT"
    elif "غير مناسب للنشر العام" in label:
        return "ADULT"
    elif "not_adult" in label or "not adult" in label:
        return "NOT_ADULT"
    elif (
        "التغريدة مناسبة للبالغين" in label
        or "المحتوى مناسب للبالغين" in label
        or "للبالغين" in label
        or "نعم" in label
        or "adult" in label
    ):
        return "ADULT"
    elif (
        "ليس" in label
        or "ليست" in label
        or "not" in label
        or "no" == label
        or "غير" in label
        or "لا" in label
    ):
        return "NOT_ADULT"
    else:
        return None
