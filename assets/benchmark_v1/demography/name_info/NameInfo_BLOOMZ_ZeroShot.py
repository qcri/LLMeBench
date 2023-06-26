import os

from arabic_llm_benchmark.datasets import NameInfoDataset
from arabic_llm_benchmark.models import BLOOMPetalModel
from arabic_llm_benchmark.tasks import DemographyNameInfoTask


def config():
    return {
        "dataset": NameInfoDataset,
        "dataset_args": {},
        "task": DemographyNameInfoTask,
        "task_args": {},
        "model": BLOOMPetalModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "class_labels": [
                "gb",
                "us",
                "cl",
                "fr",
                "ru",
                "pl",
                "in",
                "it",
                "kr",
                "gh",
                "ca",
                "sa",
                "at",
                "de",
                "cn",
                "br",
                "dk",
                "se",
                "bd",
                "cu",
                "jp",
                "be",
                "es",
                "co",
                "id",
                "iq",
                "pk",
                "tr",
                "il",
                "ch",
                "ar",
                "ro",
                "nl",
                "ps",
                "ug",
                "ir",
                "cg",
                "do",
                "ee",
                "tn",
                "gr",
                "np",
                "ie",
                "sy",
                "hu",
                "eg",
                "ma",
                "ve",
                "ph",
                "no",
                "bg",
                "si",
                "ke",
                "au",
                "et",
                "py",
                "af",
                "pt",
                "th",
                "bo",
                "mx",
                "lb",
                "za",
                "fi",
                "hr",
                "vn",
                "ly",
                "nz",
                "qa",
                "kh",
                "ci",
                "ng",
                "sg",
                "cm",
                "dz",
                "tz",
                "ae",
                "pe",
                "az",
                "lu",
                "ec",
                "cz",
                "ua",
                "uy",
                "sd",
                "ao",
                "my",
                "lv",
                "kw",
                "tw",
                "bh",
                "lk",
                "ye",
                "cr",
                "jo",
                "pa",
                "om",
                "uz",
                "by",
                "kz",
            ],
            "max_tries": 10,
        },
        "general_args": {
            "data_path": "data/demographic_attributes/name_info/wikidata_test.txt"
        },
    }


def prompt(input_sample):
    arr = input_sample.split()

    if len(arr) > 1000:
        input_sample = " ".join(arr[:1000])
    else:
        input_sample = " ".join(arr)

    prompt_string = (
        f"You are an expert annotator who can identify the country of a person based on name.\n"
        f"Label the country of the following person 'name'. Write ONLY the country code in ISO 3166-1 alpha-2 format.\n"
        f"Provide only label.\n\n"
        f"name: {input_sample}\n"
        f"country: \n"
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

    if "name: " in label:
        label_fixed = label.replace("name: ", "").lower()
    elif label.lower() in label_list:
        label_fixed = label.lower()
    elif (
        "I'm sorry, but I cannot predict the country" in label
        or "I cannot predict the country" in label
    ):
        label_fixed = None
    else:
        label_fixed = None

    return label_fixed
