from llmebench.datasets import EmotionDataset
from llmebench.models import RandomModel
from llmebench.tasks import EmotionTask, TaskType


def config():
    return {
        "dataset": EmotionDataset,
        "dataset_args": {},
        "task": EmotionTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.MultiLabelClassification,
            "class_labels": [
                "anger",
                "disgust",
                "fear",
                "joy",
                "love",
                "optimism",
                "pessimism",
                "sadness",
                "surprise",
                "trust",
            ],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    emotions_positions = {
        "anger": 0,
        "anticipation": 1,
        "disgust": 2,
        "fear": 3,
        "joy": 4,
        "love": 5,
        "optimism": 6,
        "pessimism": 7,
        "sadness": 8,
        "surprise": 9,
        "trust": 10,
    }

    results = [0] * 11

    for emotion in emotions_positions:
        if emotion in response["random_response"]:
            results[emotions_positions[emotion]] = 1
    return results
