import random

from llmebench.datasets import BibleMaghrebiDiacritizationDataset
from llmebench.models import RandomModel
from llmebench.tasks import ArabicDiacritizationTask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"WER": "1.000"},
    }


def config():
    return {
        "dataset": BibleMaghrebiDiacritizationDataset,
        "task": ArabicDiacritizationTask,
        "model": RandomModel,
        "model_args": {"task_type": TaskType.Other},
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    diacritics = [
        "\u064e",  # Fatha
        "\u064b",  # Fathatan
        "\u064f",  # Damma
        "\u064c",  # Dammatan
        "\u0650",  # Kasra
        "\u064d",  # Kasratan
        "\u0652",  # Sukun
        "\u0651",  # Shadda
        None,
    ]

    shadda_diacritics = [
        "\u064e",  # Fatha
        "\u064b",  # Fathatan
        "\u064f",  # Damma
        "\u064c",  # Dammatan
        "\u0650",  # Kasra
        "\u064d",  # Kasratan
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
