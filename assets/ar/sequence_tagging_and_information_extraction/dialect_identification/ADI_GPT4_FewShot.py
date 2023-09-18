from llmebench.datasets import ADIDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import DialectIDTask


def config():
    return {
        "dataset": ADIDataset,
        "dataset_args": {},
        "task": DialectIDTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "class_labels": [
                "IRA",
                "JOR",
                "KSA",
                "KUW",
                "LEB",
                "LIB",
                "PAL",
                "QAT",
                "SUD",
                "SYR",
                "UAE",
                "YEM",
            ],
            "max_tries": 30,
        },
        "general_args": {
            "fewshot": {"deduplicate": False, "train_split": "dev"},
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    out_prompt = out_prompt + "Here are some examples:\n\n"
    for index, example in enumerate(examples):
        out_prompt = (
            out_prompt
            + "Example "
            + str(index)
            + ":"
            + "\n"
            + "text: "
            + example["input"]
            + "\nlabel: "
            + example["label"]
            + "\n\n"
        )

    out_prompt = out_prompt + "text: " + input_sample + "\nlabel: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = f'Classify the following "text" into one of the following dialect categories: "IRA", "JOR", "KSA", "KUW", "LEB", "LIB", "PAL", "QAT", "SUD", "SYR", "UAE", "YEM"'

    return [
        {
            "role": "system",
            "content": "As an expert annotator, you have the ability to identify and annotate 'text' in different dialects.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()
    label_list = config()["model_args"]["class_labels"]
    label_list = [dialect.lower() for dialect in label_list]

    label = label.replace("label:", "").strip()

    if label in label_list:
        label_fixed = label
    elif "\n msa" in label:
        label_fixed = "msa"
    elif "\n ksa" in label:
        label_fixed = "ksa"
    elif "\n pal" in label:
        label_fixed = "pal"
    elif "\n egy" in label:
        label_fixed = "egy"
    elif "\n yem" in label:
        label_fixed = "yem"
    elif "\n syr" in label:
        label_fixed = "syr"
    elif "\n jor" in label:
        label_fixed = "jor"
    elif "\n ira" in label:
        label_fixed = "ira"
    elif "\n kuw" in label:
        label_fixed = "kuw"
    else:
        label_fixed = None

    return label_fixed