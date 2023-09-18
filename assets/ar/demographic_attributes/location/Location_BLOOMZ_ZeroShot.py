from llmebench.datasets import LocationDataset
from llmebench.models import PetalsModel
from llmebench.tasks import DemographyLocationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"Macro-F1": "0.118"},
    }


def config():
    return {
        "dataset": LocationDataset,
        "dataset_args": {},
        "task": DemographyLocationTask,
        "task_args": {},
        "model": PetalsModel,
        "model_args": {
            "class_labels": [
                "ae",
                "OTHERS",
                "bh",
                "dz",
                "eg",
                "iq",
                "jo",
                "kw",
                "lb",
                "ly",
                "ma",
                "om",
                "ps",
                "qa",
                "sa",
                "sd",
                "so",
                "sy",
                "tn",
                "UNK",
                "ye",
                "mr",
            ],
            "max_tries": 2,
        },
    }


def prompt(input_sample):
    prompt_string = (
        f"You are an expert at identifying the ISO 3166-1 alpha-2 'country code' from the user location mentioned on Twitter.\n"
        f"Given the following 'user location', identify and map it to its corresponding country code in accordance with ISO 3166-1 alpha-2. "
        f"Please write the country code only, with no additional explanations. "
        f"If the country is not an Arab country, please write 'OTHERS'. If the location doesn't map to a recognized country, write 'UNK'.\n"
        f"Provide only label.\n\n"
        f"user location: {input_sample}\n"
        f"country code: "
    )

    return {
        "prompt": prompt_string,
    }


def post_process(response):
    label = response["outputs"].strip()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")
    label = label.lower()
    label_list = config()["model_args"]["class_labels"]

    if "country code: " in label:
        label_fixed = label.replace("country code: ", "")
    elif label in label_list:
        label_fixed = label
    else:
        label_fixed = None

    return label_fixed
