import random

from llmebench.datasets import WikiNewsLemmatizationDataset
from llmebench.models import RandomModel
from llmebench.tasks import LemmatizationTask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"Accuracy": "0.348"},
    }


def config():
    return {
        "dataset": WikiNewsLemmatizationDataset,
        "task": LemmatizationTask,
        "model": RandomModel,
        "model_args": {"task_type": TaskType.Other},
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    lemmatized_sentence = []

    for token in response["random_response"].split(" "):
        if len(token) > 1:
            # Keep atleast 1 character in lemma
            nonlemma_length = random.choice(range(len(token) - 1))

            prefix_length = random.choice(range(nonlemma_length + 1))
            suffix_length = nonlemma_length - prefix_length

            token = token[prefix_length : len(token) - suffix_length]

        lemmatized_sentence.append(token)

    return (None, " ".join(lemmatized_sentence))
