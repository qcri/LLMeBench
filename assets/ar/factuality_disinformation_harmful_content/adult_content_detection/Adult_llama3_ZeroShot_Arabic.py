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


def prompt(input_sample):
    return [
        {
            "role": "user",
            "content": (
                "هل تحتوي التغريدة التالية على لغة للبالغين؟ اجب 'للبالغين' أو 'ليست للبالغين' بدون أي تفسير.\n\n"
                + "تغريدة: "
                + input_sample
                + "\nالتصنيف: "
            ),
        }
    ]


def post_process(response):
    # if not response or 'error' in response or 'output' not in response:
    # print("Error or missing output in response:", response)
    # return "No Response"

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
