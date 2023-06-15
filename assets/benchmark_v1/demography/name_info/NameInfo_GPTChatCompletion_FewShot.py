import os

from arabic_llm_benchmark.datasets import NameInfoDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import DemographyNameInfoTask


def config():
    return {
        "dataset": NameInfoDataset,
        "dataset_args": {},
        "task": DemographyNameInfoTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
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
        "general_args": {
            "data_path": "data/demographic_attributes/name_info/wikidata_test.txt",
            "fewshot": {
                "train_data_path": "data/demographic_attributes/name_info/wikidata_test.txt", # TODO need to change the file
            },
        },

    }

def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n\n"
    for example in examples:
        out_prompt = (
            out_prompt
            + "name: "
            + example["input"]
            + "\ncountry: "
            + example["label"]
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "name: " + input_sample + "\ncountry: \n"

    return out_prompt

def prompt(input_sample, examples):
    base_prompt = (
        f"Label the country of the following person 'name'. Write ONLY the country code in ISO 3166-1 alpha-2 format."
    )
    return [
        {
            "role": "system",
            "content": "You are an expert annotator who can identify the country of a person based on name.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    if("name: " in label):
        label_fixed = label.replace("name: ", "").lower()
    elif("I'm sorry, but I cannot predict the country" in label or "I cannot predict the country" in label):
        label_fixed="NameIssue"
    else:
        label_fixed = None

    return label_fixed
