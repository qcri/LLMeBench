from llmebench.datasets import NameInfoDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import DemographyNameInfoTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
    }


def config():
    return {
        "dataset": NameInfoDataset,
        "task": DemographyNameInfoTask,
        "model": OpenAIModel,
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
        "general_args": {
            "fewshot": {
                "deduplicate": False,
            },
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    out_prompt = out_prompt + "Here are some examples:\n\n"

    for index, example in enumerate(examples):
        out_prompt = (
            out_prompt
            + "Example "
            + str(index)
            + ":"
            + "\n"
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
    base_prompt = f"Label the country of the following person 'name'. Write ONLY the country code in ISO 3166-1 alpha-2 format."
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
    label = response["choices"][0]["message"]["content"].strip()
    country_dict = config()["model_args"]["class_labels"]
    if "country: " in label:
        label_fixed = label.replace("country: ", "").lower()
    elif label.lower() in country_dict:
        label_fixed = label.lower()
    elif (
        "I'm sorry, but I cannot predict the country" in label
        or "I cannot predict the country" in label
    ):
        label_fixed = "NameIssue"  # TODO need to fix this
    else:
        label_fixed = None

    return label_fixed
