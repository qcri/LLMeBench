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
            "class_labels": ["0", "1"],
            "max_tries": 30,
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    base_prompt = (
        f"هل تحتوي هذه التغريدة على ادعاء يمكن التحقق منه؟ أجب فقط بـ 'نعم' أو 'لا'.\n\n"
        f"التغريدة: {input_sample}\n"
        f"التصنيف: \n"
    )
    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label = label.replace("التصنيف:", "").strip()
    label = label.lower()

    if "لا يمكنني" in label:
        return None
    if "التصنيف: " in label:
        arr = label.split("التصنيف: ")
        label = arr[1].strip()

    if "نعم" in label:
        label_fixed = "1"
    elif "لا" in label:
        label_fixed = "0"
    else:
        label_fixed = None

    return label_fixed
