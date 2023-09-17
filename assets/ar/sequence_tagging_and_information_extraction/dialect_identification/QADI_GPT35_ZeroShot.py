from llmebench.datasets import QADIDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import DialectIDTask


def config():
    return {
        "dataset": QADIDataset,
        "dataset_args": {},
        "task": DialectIDTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "class_labels": [
                "EG",
                "DZ",
                "SD",
                "YE",
                "SY",
                "AE",
                "JO",
                "LY",
                "PS",
                "OM",
                "QA",
                "BH",
                "MSA",
                "SA",
                "IQ",
                "MA",
            ],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    prompt_string = (
        f'Write only the country code of the Arabic country in which this sentence is written in its dialect without any explanation. Write only the country code in ISO 3166-1 alpha-2 format without explanation. Write "MSA" if the sentence is written in Modern Standard Arabic.\n'
        f"Please provide only the label.\n\n"
        f"text: {input_sample}\n"
        f"label: \n"
    )

    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": prompt_string,
            }
        ],
    }


def post_process(response):
    label = response["choices"][0]["text"]

    label_list = config()["model_args"]["class_labels"]
    label_list = [dialect for dialect in label_list]

    label = label.replace("label:", "").strip()

    # j = out.find(".")
    # if j > 0:
    #     out = out[0:j]

    if label in label_list:
        label_fixed = label
    else:
        label_fixed = None

    return label_fixed
