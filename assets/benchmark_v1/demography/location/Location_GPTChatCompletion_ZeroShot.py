import os

from arabic_llm_benchmark.datasets import LocationDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import DemographyLocationTask


def config():
    return {
        "dataset": LocationDataset,
        "dataset_args": {},
        "task": DemographyLocationTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
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
        "general_args": {
            "data_path": "data/demographic_attributes/location/arab+others.txt"
        },
    }


def prompt(input_sample):
    prompt_string = (
        f"Given the following 'user location', identify and map it to its corresponding 'country code' in accordance with ISO 3166-1 alpha-2. "
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
    label = response["choices"][0]["message"]["content"].lower().strip()
    country_code_list = config()["model_args"]["class_labels"]
    if "country code: " in label:
        # label_fixed = label.replace("country code: ", "").strip()
        label = label.split("country code: ")[1].strip()

    if label in country_code_list:
        label_fixed = label
    elif label == "unk":
        label_fixed = "UNK"
    elif label == "others":
        label_fixed = "OTHERS"
    else:
        label_fixed = None

    return label_fixed
