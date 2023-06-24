import os

from arabic_llm_benchmark.datasets import QADIDataset
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel
from arabic_llm_benchmark.tasks import DialectIDTask


def config():
    return {
        "dataset": QADIDataset,
        "dataset_args": {},
        "task": DialectIDTask,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
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
        "general_args": {
            "data_path": "data/sequence_tagging_ner_pos_etc/dialect_identification/QADI_test-PalestinePS-corrected.txt"
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
