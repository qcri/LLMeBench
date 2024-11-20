from llmebench.datasets import OSACT4SubtaskBDataset
from llmebench.models import FastChatModel
from llmebench.tasks import HateSpeechTask


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "jais-13b-chat",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": OSACT4SubtaskBDataset,
        "task": HateSpeechTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["HS", "NOT_HS"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "user",
            "content": (
                "صنف التغريدة التالية جسب احتوائها على خطاب كراهية أجب ب 'كراهية' أو 'لا كراهية' وقدم التصنيف فقط.\n\n"
                + "التغريدة: "
                + input_sample
                + "التصنيف: "
            ),
        }
    ]


def post_process(response):
    print(response)
    label = response["choices"][0]["message"]["content"].lower()
    if "لا يمكنني" in label:
        return None
    if (
        "لا كراهية" in label
        or "لا تحتوي" in label
        or "ليست كراهية" in label
        or "ليس" in label
        or "ليست" in label
        or "not" in label
        or "لا" in label
    ):
        return "NOT_HS"
    if "تحتوي على خطاب كراهية" in label:
        return "HS"
    if "not" in label or "not_hs" in label or "لا كراهية" in label:
        return "NOT_HS"
    elif "hate speech" in label or "hs" in label or "كراهية" in label:
        return "HS"
    else:
        print("No clear label found.")
        return None
