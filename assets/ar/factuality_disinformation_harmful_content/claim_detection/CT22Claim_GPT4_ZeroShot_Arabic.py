from llmebench.datasets import CT22ClaimDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ClaimDetectionTask


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "GPT-4o-2024-05-22",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
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
        f"هل تحتوي هذه التغريدة على ادعاء؟ أجب فقط بـ 'نعم' أو 'لا'. قدم التصنيف فقط.\n"
        f"التغريدة: {input_sample}\n"
        f"التصنيف: \n"
    )
    return [
        {
            "role": "system",
            "content": "أنت خبير في تحليل و تصنيف التغريدات.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    input_label = response["choices"][0]["message"]["content"]
    input_label = input_label.replace(".", "").strip().lower()

    if (
        "لا" in input_label
        or "لا تحتوي" in input_label
        or "ليست" in input_label
        or "not" in input_label
        or "label: 0" in input_label
        or "label: no" in input_label
        or "not contain" in input_label
        or "doesn't contain" in input_label
    ):
        return "0"

    elif (
        "نعم" in input_label
        or "تحتوي" in input_label
        or "yes" in input_label
        or "contains" in input_label
        or "label: 1" in input_label
    ):
        return "1"
    else:
        return None
