import re

from llmebench.datasets import QCRIDialectalArabicSegmentationDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import ArabicSegmentationTask


def config():
    sets = [
        ("egy", "egy.seg/egy.data_5.test.src.sent"),
        ("glf", "glf.seg/glf.data_5.test.src.sent"),
        ("mgr", "mgr.seg/mgr.data_5.test.src.sent"),
        ("lev", "lev.seg/lev.data_5.test.src.sent"),
    ]
    configs = []
    for name, testset in sets:
        configs.append(
            {
                "name": name,
                "config": {
                    "dataset": QCRIDialectalArabicSegmentationDataset,
                    "dataset_args": {},
                    "task": ArabicSegmentationTask,
                    "task_args": {},
                    "model": LegacyOpenAIModel,
                    "model_args": {
                        "max_tries": 3,
                    },
                    "general_args": {
                        "data_path": f"data/sequence_tagging_ner_pos_etc/segmentation/{testset}"
                    },
                },
            }
        )
    return configs


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
