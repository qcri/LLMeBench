from llmebench.datasets import CT22ClaimDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ClaimDetectionTask





def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "GPT-4o-2024-05-22",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }





def config():
    return {
        "dataset": CT22ClaimDataset,
        "task": ClaimDetectionTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["0", "1"],
            "max_tries": 30,
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    prompt_string = (
        f"Does the following tweet contain a factual claim? If it does, return 'yes', if it does not, return 'no'. Provide only label.\n\n"
        f"tweet: {input_sample}\n"
        f"label: \n"
    )
    return [
        {
            "role": "system",
            "content": "You are expert in text analysis and classification.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    input_label = response["choices"][0]["message"]["content"]
    input_label = input_label.replace(".", "").strip().lower()
    pred_label = ""

    if (
        "yes" in input_label
        or "contains a factual claim" in input_label
        or "label: 1" in input_label
    ):
        pred_label = "1"
    if (
        input_label == "no"
        or "label: 0" in input_label
        or "label: no" in input_label
        or "not contain" in input_label
        or "doesn't contain" in input_label
    ):
        pred_label = "0"

    if pred_label == "":
        pred_label = None

    return pred_label
