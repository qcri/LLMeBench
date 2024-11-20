from llmebench.datasets import SpamDataset
from llmebench.models import AzureModel
from llmebench.tasks import SpamTask


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "Llama-3.1-8B-Instruct",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": SpamDataset,
        "task": SpamTask,
        "model": AzureModel,
        "model_args": {
            "class_labels": ["__label__ADS", "__label__NOTADS"],
            "max_tries": 100,
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n\n"
    out_prompt += "إليك بعض الأمثلة:\n\n"
    for index, example in enumerate(examples):
        label = "إعلان" if example["label"] == "__label__ADS" else "ليس إعلان"
        out_prompt += (
            f"مثال {index + 1}:\n"
            f"التغريدة: '{example['input']}'\n"
            f"التصنيف: {label}\n\n"
        )

    # Append the sentence we want the model to predict for but leave the label blank
    out_prompt += f"التغريدة: '{input_sample}'\nالتصنيف: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = (
        "هل تحتوي التغريدة التالية على محتوى إعلان؟\n"
        "حدد التصنيف المناسب بناءً على المحتوى:\n"
        " - إذا كانت التغريدة تحتوي على إعلان، أجب بـ 'إعلان'.\n"
        " - إذا كانت التغريدة لا تحتوي على إعلان، أجب بـ 'ليس إعلان'.\n"
        "قدم التصنيف فقط.\n"
    )
    return [
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        }
    ]


def post_process(response):
    if "output" in response:
        label = response["output"].strip()
        label = label.replace("<s>", "")
        label = label.replace("</s>", "")
        label = label.lower()
    else:
        print("Error or missing output in response:", response)
        return None

    label = label.replace("التصنيف:", "").strip().lower()
    label = label.replace("label:", "").strip().lower()

    # print("label", label)
    if "لا أستطيع" in label or "I cannot" in label:
        return None
    if (
        "ليست" in label
        or "not" in label
        or "غير" in label
        or "no" in label
        or "ليس" in label
    ):
        return "__label__NOTADS"
    elif (
        "نعم" in label
        or "إعلان" in label
        or "spam" in label
        or "مزعج" in label
        or "اعلان" in label
        or "مرغوب" in label
        or "غير" in label
    ):
        return "__label__ADS"
    else:
        return None
