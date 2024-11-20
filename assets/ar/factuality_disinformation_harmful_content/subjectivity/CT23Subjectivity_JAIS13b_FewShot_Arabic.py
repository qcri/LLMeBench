from llmebench.datasets import CT23SubjectivityDataset
from llmebench.models import FastChatModel
from llmebench.tasks import SubjectivityTask


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "jais-13b-chat",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": CT23SubjectivityDataset,
        "task": SubjectivityTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["SUBJ", "OBJ"],
            "max_tries": 30,
        },
        "general_args": {"test_split": "ar/dev", "fewshot": {"train_split": "ar"}},
    }


def prompt(input_sample, examples):
    base_prompt = ' صنف التغريدة الى "موضوعية" أو "غير موضوعية". قم بتقديم التصنيف فقط لأخر تغريدة، لا تقدم أي تبرير إضافي.:\n'
    return [
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    out_prompt = out_prompt + "هذه بعض الأمثلة:\n\n"
    for index, example in enumerate(examples):
        label = "موضوعية" if example["label"] == "OBJ" else "غير موضوعية"

        out_prompt = (
            out_prompt
            + "مثال "
            + str(index)
            + ":"
            + "\n"
            + "التغريدة: "
            + example["input"]
            + "\التصنيف: "
            + label
            + "\n\n"
        )

    out_prompt = out_prompt + "التغريدة: " + input_sample + "\التصنيف: \n"

    return out_prompt


def post_process(response):
    label = (
        response["choices"][0]["message"]["content"].lower().replace(".", "").strip()
    )

    if "غير" in label or "subj" in label or "not" in label or "ليس" in label:
        return "SUBJ"
    elif "موضوعية" in label or "obj" in label:
        return "OBJ"
    else:
        return None
