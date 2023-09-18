import random

from llmebench.datasets import BibleMaghrebiDiacritizationDataset
from llmebench.models import RandomModel
from llmebench.tasks import ArabicDiacritizationTask, TaskType


def config():
    return {
        "dataset": BibleMaghrebiDiacritizationDataset,
        "dataset_args": {},
        "task": ArabicDiacritizationTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {"task_type": TaskType.Other},
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    diacritics = [
        "\u064E",  # Fatha
        "\u064B",  # Fathatan
        "\u064F",  # Damma
        "\u064C",  # Dammatan
        "\u0650",  # Kasra
        "\u064D",  # Kasratan
        "\u0652",  # Sukun
        "\u0651",  # Shadda
        None,
    ]

    shadda_diacritics = [
        "\u064E",  # Fatha
        "\u064B",  # Fathatan
        "\u064F",  # Damma
        "\u064C",  # Dammatan
        "\u0650",  # Kasra
        "\u064D",  # Kasratan
        None,
    ]

    diacritized_sentence = []
    for token in response["random_response"].split(" "):
        new_token = []
        for character in token:
            new_token.append(character)

            diacritic = random.choice(diacritics)
            if diacritic is None:
                continue

            if diacritic is diacritics[-2]:  # Shadda
                extra_diacritic = random.choice(shadda_diacritics)
                if extra_diacritic:
                    new_token.append(extra_diacritic)
                new_token.append(diacritic)
        diacritized_sentence.append("".join(new_token))

    return " ".join(diacritized_sentence)
