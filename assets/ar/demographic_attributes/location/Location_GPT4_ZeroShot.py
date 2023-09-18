from llmebench.datasets import LocationDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import DemographyLocationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Macro-F1": "0.735"},
    }


def config():
    return {
        "dataset": LocationDataset,
        "dataset_args": {},
        "task": DemographyLocationTask,
        "task_args": {},
        "model": OpenAIModel,
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
    prompt_string = (
        f"Given the following 'user location', identify and map it to its corresponding country code in accordance with ISO 3166-1 alpha-2. "
        f"Please write the country code only, with no additional explanations. "
        f"If the country is not an Arab country, please write 'OTHERS'. If the location doesn't map to a recognized country, write 'UNK'.\n\n"
        f"user location: {input_sample}\n"
        f"country code: \n"
    )

    return [
        {
            "role": "system",
            "content": "You are an expert at identifying the ISO 3166-1 alpha-2 country code from the user location mentioned on Twitter.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()

    label_list = config()["model_args"]["class_labels"]

    if "country code: " in label:
        label_fixed = label.replace("country code: ", "")
    elif label in label_list:
        label_fixed = label
    else:
        label_fixed = None

    return label_fixed
