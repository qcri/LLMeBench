import os

from llmebench.datasets import QADIDataset
from llmebench.models import BLOOMPetalModel
from llmebench.tasks import DialectIDTask

def config():
    return {
        "dataset": QADIDataset,
        "dataset_args": {},
        "task": DialectIDTask,
        "task_args": {},
        "model": BLOOMPetalModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "class_labels": [
                "EG",
                "DZ",
                "SD",
                "YE",
                "SY",
                "TN",
                "AE",
                "JO",
                "LY",
                "PS",
                "OM",
                "LB",
                "KW",
                "QA",
                "BH",
                "MSA",
                "SA",
                "IQ",
                "MA",
            ],
            "max_tries": 0,
        },
        "general_args": {
            "data_path": "data/sequence_tagging_ner_pos_etc/dialect_identification/QADI_test-PalestinePS-corrected.txt",
        },
    }


def prompt(input_sample):
    prompt_string = (
        f'Identify the dialect of the following Arabic "text" given the following possible list of dialects: “Egyptian”, “Algerian”,  "Sudanese",  "Yemeni",  "Syrian",  “Tunisian”, "Emirati",  "Jordanian",  "Libyan",  "Palestinian",  "Omani",  “Lebanese”, “Kuwaiti”, "Qatari",  "Bahrani",  "modern standard Arabic",  "Saudi",  "Iraqi",  "Moroccan"\n\n'
        f"text: {input_sample}\n"
        f"label: \n"
    )

    return {
        "prompt": prompt_string,
    }


def post_process(response):
    count_label_map = {
        "Egyptian": "EG",
        "Algerian": "DZ",
        "Sudanese": "SD",
        "Yemeni": "YE",
        "Syrian": "SY",
        "Tunisian": "TN",
        "Emirati": "AE",
        "Jordanian": "JO",
        "Libyan": "LY",
        "Palestinian": "PS",
        "Omani": "OM",
        "Lebanese": "LB",
        "Kuwaiti": "KW",
        "Qatari": "QA",
        "Bahrani": "BH",
        "modern standard Arabic": "MSA",
        "Modern standard Arabic": "MSA",
        "Modern Standard Arabic": "MSA",
        "Saudi": "SA",
        "Iraqi": "IQ",
        "Moroccan": "MA",
    }

    label = response["outputs"].strip()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")
    label = label.replace("Dialect: ", "").replace("dialect: ","")
    label = label.replace("label: ", "")
    label = label.strip()

    if label in count_label_map:
        label_fixed = count_label_map[label]
    else:
        label_fixed = None

    return label_fixed
