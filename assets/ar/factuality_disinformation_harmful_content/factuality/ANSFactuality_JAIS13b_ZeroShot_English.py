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
    base_prompt = (
        "Detect whether the information in the sentence is factually true or false. "
        "Answer only by true or false.\n\n"
        + "Sentence: "
        + input_sample
        + "\nlabel: \n"
    )

    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    input_label = response["choices"][0]["message"]["content"]
    input_label = input_label.replace(".", "").strip().lower()

    if (
        "true" in input_label
        or "yes" in input_label
        or "label: 1" in input_label
        or "label: yes" in input_label
    ):
        pred_label = "true"
    elif (
        "false" in input_label
        or "label: 0" in input_label
        or "label: no" in input_label
    ):
        pred_label = "false"
    else:
        print("label problem!! " + input_label)
        pred_label = None

    return pred_label
