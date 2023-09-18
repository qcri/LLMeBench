from llmebench.datasets import ADIDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import DialectIDTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Macro-F1": "0.229"},
    }


def config():
    return {
        "dataset": ADIDataset,
        "task": DialectIDTask,
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
    }


def prompt(input_sample):
    prompt_string = (
        f'Classify the following "text" into one of the following dialect categories: "IRA", "JOR", "KSA", "KUW", "LEB", "LIB", "PAL", "QAT", "SUD", "SYR", "UAE", "YEM"\n'
        f"Please provide only the label.\n\n"
        f"text: {input_sample}\n"
        f"label: \n"
    )

    return [
        {
            "role": "system",
            "content": "As an expert annotator, you have the ability to identify and annotate 'text' in different dialects.",
        },
        {
            "role": "user",
            "content": prompt_string,
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
