from llmebench.datasets import CT22ClaimDataset
from llmebench.models import FastChatModel
from llmebench.tasks import ClaimDetectionTask


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "jais-13b-chat",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": CT22ClaimDataset,
        "task": ClaimDetectionTask,
        "model": FastChatModel,
        "model_args": {
            "max_tries": 3,
        },
        "general_args": {"test_split": "ar", "fewshot": {"train_split": "ar"}},
    }


def prompt(input_sample, examples=None):
    base_prompt = "Does this sentence contain a factual claim? Answer with 'yes' or 'no' only. Provide only the label.\n"
    if examples:
        user_message_content = few_shot_prompt(input_sample, base_prompt, examples)
    else:
        user_message_content = base_prompt + f"Sentence: {input_sample}\nLabel: "

    return [{"role": "user", "content": user_message_content}]


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    for example in examples:
        label = "no" if example["label"] == "0" else "yes"
        out_prompt += "Sentence: " + example["input"] + "\nLabel: " + label + "\n\n"
    out_prompt += "Sentence: " + input_sample + "\nLabel: "

    return out_prompt


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label = label.replace("label:", "").strip()
    label = label.lower()

    if (
        "لا يمكنني" in label
        or "I cannot" in label
        or "sorry" in label
        or "هذه المحادثة غير مناسبة" in label
    ):
        return None
    if "هذه التغريدة تحتوي" in label:
        return "1"

    if (
        "not a factual claim" in label
        or "لا يوجد" in label
        or "not" in label
        or "لا" in label
    ):
        return "0"
    return "1"

    if "label: " in label:
        arr = label.split("label: ")
        label = arr[1].strip()
        if "yes" in label:
            pred_label = "1"
        elif "no" in label:
            pred_label = "0"
        else:
            pred_label = "0"

        print(f"Predicted Label: {pred_label}")

        return pred_label
