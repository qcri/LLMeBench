from llmebench.datasets import OSACT4SubtaskBDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import HateSpeechTask


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "GPT-4o-2024-05-22",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": OSACT4SubtaskBDataset,
        "task": HateSpeechTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["HS", "NOT_HS"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "أنت خبير في تحليل و تصنيف التغريدات.",
        },
        {
            "role": "user",
            "content": f' هل تحتوي التغريدة التالية على خطاب كراهية؟ أجب بـ "كراهية" إن احتوت على خطاب كراهية، و أجب بـ "ليس كراهية" إن لم تكن كذلك. \n'
            + "التغريدة: "
            + input_sample
            + "\n"
            + "التصنيف: ",
        },
    ]


def post_process(response):
    out = response["choices"][0]["message"]["content"]
    label = out.lower().strip()

    if (
        "ليس" in label
        or "ليس كراهية" in label
        or "لا" in label
        or "no" in label
        or "not" in label
        or "don't" in label
        or "not_hs" in label
        or "not_hatespeech" in label
        or "not_hate_speech" in label
    ):
        return "NOT_HS"
    elif (
        "كراهية" in label
        or "نعم" in label
        or "أجل" in label
        or "yes" in label
        or "contins" in label
        or "hs" in label
        or "hatespeech" in label
        or "hate speech" in label
    ):
        return "HS"
    else:
        return None
