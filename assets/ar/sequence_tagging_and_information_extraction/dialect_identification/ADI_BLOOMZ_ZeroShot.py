from llmebench.datasets import ADIDataset
from llmebench.models import PetalsModel
from llmebench.tasks import DialectIDTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"Macro-F1": "0.098"},
    }


def config():
    return {
        "dataset": ADIDataset,
        "task": DialectIDTask,
        "model": PetalsModel,
    }


def prompt(input_sample):
    prompt_string = (
        f'Identify the dialect of the following Arabic "text" given the following possible dialects: "Egyptian", "Iraqi", "Jordanian", "Saudi", "Kuwaiti", "Lebanese", "Libyan", "Moroccan", "modern standard Arabic", "Palestinian", "Qatari", "Sudanese", "Syrian", "Emirati", "Yemeni"\n'
        f"text: {input_sample}\n"
        f"label: \n"
    )

    return {
        "prompt": prompt_string,
    }


def post_process(response):
    count_label_map = {
        "Egyptian": "EGY",
        "Iraqi": "IRA",
        "Jordanian": "JOR",
        "Saudi": "KSA",
        "Kuwaiti": "KUW",
        "Lebanese": "LEB",
        "Libyan": "LIB",
        "Moroccan": "MOR",
        "modern standard Arabic": "MSA",
        "Modern standard Arabic": "MSA",
        "Modern Standard Arabic": "MSA",
        "Palestinian": "PAL",
        "Qatari": "QAT",
        "Sudanese": "SUD",
        "Syrian": "SYR",
        "Emirati": "UAE",
        "Yemeni": "YEM",
        "Yemen": "YEM",
    }

    label = response["outputs"].strip()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")
    label = label.replace("Dialect: ", "").replace("dialect: ", "")
    label = label.replace("label: ", "")
    label = label.strip()

    if label in count_label_map:
        label_fixed = count_label_map[label].lower()
    else:
        label_fixed = None

    return label_fixed
