from llmebench.datasets import NameInfoDataset
from llmebench.models import FastChatModel
from llmebench.tasks import DemographyNameInfoTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "JAIS-13b",
        "description": "Locally hosted JAIS-13b-chat model using FastChat.",
        "scores": {"Macro-F1": ""},
    }


def config():
    return {
        "dataset": NameInfoDataset,
        "task": DemographyNameInfoTask,
        "model": FastChatModel,
        "model_args": {
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
            "max_tries": 30,
        },
    }


def prompt(input_sample):
    base_prompt = (
        f"Label the country of the following person 'name'. Write ONLY the country code in ISO 3166-1 alpha-2 format.\n\n"
        f"name: {input_sample}\n"
        f"country: \n"
    )
    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label_list = config()["model_args"]["class_labels"]

    if "name: " in label:
        label_fixed = label.replace("name: ", "").lower()
    elif label.lower() == 'uae':
        label_fixed = 'ae'
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
