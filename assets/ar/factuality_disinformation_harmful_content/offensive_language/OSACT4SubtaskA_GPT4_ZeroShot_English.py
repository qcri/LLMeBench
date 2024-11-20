from llmebench.datasets import OSACT4SubtaskADataset
from llmebench.models import OpenAIModel
from llmebench.tasks import OffensiveTask


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "GPT-4o-2024-05-22",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": OSACT4SubtaskADataset,
        "task": OffensiveTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["OFF", "NOT_OFF"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are an expert in tweet analysis and classification.",
        },
        {
            "role": "user",
            "content": (
                "If the following tweet is offensive, label it as 'offensive'. Return 'not offensive' otherwise."
                + "\n\ntweet: "
                + input_sample
                + "\nlabel: "
            ),
        },
    ]


def post_process(response):
    out = response["choices"][0]["message"]["content"]
    label = out.strip().lower()
    if "غير" in label or "not" in label or "ليس" in label or "ليست" in label:
        return "NOT_OFF"
    elif (
        "مسيء" in label or "offensive" in label or "المساءاة" in label or "off" in label
    ):
        return "OFF"
    else:
        return None
