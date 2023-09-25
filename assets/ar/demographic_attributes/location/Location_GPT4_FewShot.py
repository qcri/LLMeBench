from llmebench.datasets import LocationDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import DemographyLocationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
    }


def config():
    return {
        "dataset": LocationDataset,
        "task": DemographyLocationTask,
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
            + "user location: "
            + example["input"]
            + "\ncountry code: "
            + example["label"]
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "user location: " + input_sample + "\ncountry code: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = f"Given the following 'user location', identify and map it to its corresponding country code in accordance with ISO 3166-1 alpha-2. Please write the country code only, with no additional explanations. If the country is not an Arab country, please write 'OTHERS'. If the location doesn't map to a recognized country, write 'UNK'."
    return [
        {
            "role": "system",
            "content": "You are an expert at identifying the ISO 3166-1 alpha-2 country code from the user location mentioned on Twitter.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]  # .lower()
    country_code_list = config()["model_args"]["class_labels"]
    if "country code: ":
        label_fixed = label.replace("country code: ", "")
    elif label in country_code_list:
        label_fixed = label
    else:
        label_fixed = None

    return label_fixed
