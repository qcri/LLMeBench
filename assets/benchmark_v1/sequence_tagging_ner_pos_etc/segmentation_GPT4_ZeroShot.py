import os
import re

from arabic_llm_benchmark.datasets import ArabicSegmentationDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import ArabicSegmentationTask


def config():
    sets = [
        ("egy", "egy.seg/egy.data_5.test.src.sent"),
        ("glf", "glf.seg/glf.data_5.test.src.sent"),
        ("mgr", "mgr.seg/mgr.data_5.test.src.sent"),
        ("lev", "lev.seg/lev.data_5.test.src.sent"),
        ("msa", "WikiNewsTruth.txt"),
    ]
    configs = []
    for name, testset in sets:
        configs.append(
            {
                "name": name,
                "config": {
                    "dataset": ArabicSegmentationDataset,
                    "dataset_args": {},
                    "task": ArabicSegmentationTask,
                    "task_args": {},
                    "model": GPTChatCompletionModel,
                    "model_args": {
                        "api_type": "azure",
                        "api_version": "2023-03-15-preview",
                        "api_base": os.environ["AZURE_API_URL"],
                        "api_key": os.environ["AZURE_API_KEY"],
                        "engine_name": os.environ["ENGINE_NAME"],
                        # "class_labels": ["m", "f"],
                        "max_tries": 3,
                    },
                    "general_args": {
                        "data_path": "data/sequence_tagging_ner_pos_etc/segmentation/"
                        + testset
                    },
                },
            }
        )
    return configs


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are a linguist that helps in annotating data.",
        },
        {
            "role": "user",
            "content": f"A word can be composed of one root and one or multiple affixed, \
                    segment the following sentence into its morphological constituents:\n {input_sample}\
                    The input will be a list of words in the sentence. \
                    The output format should be a list of tuples, where each tuple consists of a word from the input text and its segmented form joined by a + sign.\
                    ",
        },
    ]


def post_process(response):
    results = []
    text = response["choices"][0]["message"]["content"]
    pattern = "\([\"']([^\"']+)[\"'], [\"']([^\"']+)[\"']\)"
    matches = re.finditer(pattern, text)
    for m in matches:
        results.append(m.group(2))
    # remove non-Arabic words. They don't need to be segmented

    return " ".join(results)
