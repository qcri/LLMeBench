from llmebench.datasets import ANSFactualityDataset
from llmebench.models import FastChatModel
from llmebench.tasks import FactualityTask


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "jais-13b-chat",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": ANSFactualityDataset,
        "task": FactualityTask,
        "model": FastChatModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    prompt_text = (
        "هل المعلومات في الجملة التالية صحيحة أم لا؟ أجب فقط بـ 'true' إذا كانت صحيحة و'false' إذا لم تكن صحيحة. "
        + "الجملة: "
        + input_sample
        + "\nالتصنيف: \n"
    )

    return [
        {
            "role": "user",
            "content": prompt_text,
        }
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]
    label = label.replace(".", "").strip().lower()

    if (
        "لا" in label
        or "خطأ" in label
        or "ليست" in label
        or "false" in label
        or "label: 0" in label
        or "label: no" in label
    ):
        pred_label = "false"
    elif (
        "نعم" in label
        or "صحيحة" in label
        or "true" in label
        or "label: 1" in label
        or "label: yes" in label
    ):
        pred_label = "true"
    else:
        print("label problem!! " + label)
        pred_label = None

    return pred_label
