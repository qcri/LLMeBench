import os

from arabic_llm_benchmark.datasets import DialectADIDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel, RandomGPTModel
from arabic_llm_benchmark.tasks import DialectIDTask


def config():
    return {
        "dataset": DialectADIDataset,
        "dataset_args": {},
        "task": DialectIDTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
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
            "data_path": "data/sequence_tagging_ner_pos_etc/dialect_identification/dialect_12_test_merged.tsv"
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
    else:
        label_fixed = None

    return label_fixed
