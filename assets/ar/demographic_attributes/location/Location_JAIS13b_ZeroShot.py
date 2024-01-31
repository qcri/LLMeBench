from llmebench.datasets import LocationDataset
from llmebench.models import FastChatModel
from llmebench.tasks import DemographyLocationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Jais-13b-chat",
        "description": "Locally hosted Jais-13b-chat model using FastChat.",
    }


def config():
    return {
        "dataset": LocationDataset,
        "task": DemographyLocationTask,
        "model": FastChatModel,
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
            "max_tries": 30,
        },
    }


def prompt(input_sample):
    base_prompt = (
        f"Given the following 'user location', identify and map it to its corresponding country code in accordance with ISO 3166-1 alpha-2. "
        f"Please write the country code only, with no additional explanations. "
        f"If the country is not an Arab country, please write 'OTHERS'. If the location doesn't map to a recognized country, write 'UNK'.\n\n"
        f"user location: {input_sample}\n"
        f"country code: \n"
    )

    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()

    label_list = config()["model_args"]["class_labels"]

    if "country code: " in label:
        label_fixed = label.replace("country code: ", "")
    elif label.lower() == "uae":
        label_fixed = "ae"
    elif label in label_list:
        label_fixed = label
    else:
        label_fixed = None

    return label_fixed
