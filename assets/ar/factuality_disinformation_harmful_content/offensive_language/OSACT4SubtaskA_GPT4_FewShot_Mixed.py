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


def prompt(input_sample, examples):
    base_prompt = (
        "إذا كانت التغريدة التالية تحتوي على أي درجة من الإساءة، قم بالرد بـ  'offensive' إذا لم تحتوي على أي محتوى مسيء أو مهين أو ضار، قم بالرد بـ 'not offensive'. هذه بعض الأمثلة: "
        + "\n"
    )

    return [
        {
            "role": "system",
            "content": "أنت خبير في تحليل وتصنيف التغريدات.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    for example in examples:
        # Found chatgpt confused when using 0 and 1 in the prompt
        # label = "no" if example["label"] == "0" else "yes"
        label = "not offensive" if example["label"] == "NOT_OFF" else "offensive"

        out_prompt = (
            out_prompt
            + "التغريدة: "
            + example["input"]
            + "\nالتصنيف: "
            + label
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "التغريدة: " + input_sample + "التصنيف: \n"

    # print("=========== FS Prompt =============\n")
    # print(out_prompt)

    return out_prompt


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
