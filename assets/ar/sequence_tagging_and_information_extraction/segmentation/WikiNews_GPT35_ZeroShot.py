import re

from llmebench.datasets import WikiNewsSegmentationDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import ArabicSegmentationTask


def config():
    return {
        "dataset": WikiNewsSegmentationDataset,
        "dataset_args": {},
        "task": ArabicSegmentationTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sequence_tagging_ner_pos_etc/segmentation/WikiNewsTruth.txt"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are a linguist that helps in annotating data.",
        "messages": [
            {
                "sender": "user",
                "text": f"A word can be composed of one root and one or multiple affixed, \
                    segment the following sentence into its morphological constituents:\n {input_sample}\
                    The input will be a list of words in the sentence. \
                    The output format should be a list of tuples, where each tuple consists of a word from the input text and its segmented form joined by a + sign.\
                    ",
            }
        ],
    }


def post_process(response):
    results = []
    text = response["choices"][0]["text"]
    pattern = "\([\"']([^\"']+)[\"'], [\"']([^\"']+)[\"']\)"
    matches = re.finditer(pattern, text)
    for m in matches:
        results.append(m.group(2))
    text = " ".join(results)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text
