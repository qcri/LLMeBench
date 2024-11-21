from llmebench.datasets import AdultDataset
from llmebench.models import FastChatModel
from llmebench.tasks import AdultTask


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "jais-13b-chat",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
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
    base_prompt = "Given the following tweet, label it as 'ADULT' or 'NOT_ADULT' based on the content of the tweet. Provide only label.\n\n"
    for index, example in enumerate(examples):
        label = "adult" if example["label"] == "ADULT" else "not adult"
        base_prompt += (
            f"Example {index + 1}:\n"
            f"Tweet: '{example['input']}'\n"
            f"Label: {label}\n\n"
        )
    base_prompt += f"Now classify the new tweet:\nTweet: '{input_sample}'\n" f"Label: "
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
