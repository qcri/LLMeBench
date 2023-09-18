import random

from llmebench.datasets import WikiNewsSegmentationDataset
from llmebench.models import RandomModel
from llmebench.tasks import ArabicSegmentationTask, TaskType


def config():
    return {
        "dataset": WikiNewsSegmentationDataset,
        "dataset_args": {},
        "task": ArabicSegmentationTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {"task_type": TaskType.Other},
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    segmented_sentence = []
    for token in response["random_response"].split(" "):
        if len(token) > 1:
            num_segments = min(len(token) - 1, random.choice([0, 1, 2]))
            for segment_idx, segment_loc in enumerate(
                sorted(random.sample(range(len(token) - 1), k=num_segments))
            ):
                token = (
                    token[: segment_idx + segment_loc + 1]
                    + "+"
                    + token[segment_idx + segment_loc + 1 :]
                )
        segmented_sentence.append(token)

    return " ".join(segmented_sentence)
