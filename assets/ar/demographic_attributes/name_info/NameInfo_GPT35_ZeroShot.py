from llmebench.datasets import NameInfoDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import DemographyNameInfoTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-35-turbo (version 0301)",
        "description": "GPT35 model hosted on Azure, using the Completion API. API version '2023-03-15-preview'.",
        "scores": {"Weighted-F1": "0.570"},
    }


def config():
    return {
        "dataset": NameInfoDataset,
        "task": DemographyNameInfoTask,
        "model": LegacyOpenAIModel,
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
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Predict the country of citizenship of the following person name. Write ONLY the country code in ISO 3166-1 alpha-2 format without explananation.\n {input_sample}",
            }
        ],
    }


def post_process(response):
    out = response["choices"][0]["text"]
    return out.lower()
