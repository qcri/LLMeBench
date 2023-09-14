import re

from llmebench.datasets import QCRIDialectalArabicSegmentationDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ArabicSegmentationTask


def config():
    sets = [
        ("egy", "egy.seg/egy.data_5.test.src.sent", "egy.seg/egy.data_5.dev.src.sent"),
        ("glf", "glf.seg/glf.data_5.test.src.sent", "glf.seg/glf.data_5.dev.src.sent"),
        ("mgr", "mgr.seg/mgr.data_5.test.src.sent", "mgr.seg/mgr.data_5.dev.src.sent"),
        ("lev", "lev.seg/lev.data_5.test.src.sent", "lev.seg/lev.data_5.dev.src.sent"),
    ]

    configs = []
    for name, testset, devset in sets:
        configs.append(
            {
                "name": name,
                "config": {
                    "dataset": QCRIDialectalArabicSegmentationDataset,
                    "dataset_args": {},
                    "task": ArabicSegmentationTask,
                    "task_args": {},
                    "model": OpenAIModel,
                    "model_args": {
                        "max_tries": 3,
                    },
                    "general_args": {
                        "data_path": "data/sequence_tagging_ner_pos_etc/segmentation/"
                        + testset,
                        "fewshot": {
                            "train_data_path": "data/sequence_tagging_ner_pos_etc/segmentation/"
                            + devset
                        },
                    },
                },
            }
        )
    return configs


def few_shot_prompt(input_sample, base_prompt, examples):
    output_prompt = base_prompt + "\n"
    for example in examples:
        tokens = example["input"]
        label = example["label"]
        output_prompt = output_prompt + f"Sentence: {tokens}\nLabels: {label}\n"
    output_prompt = output_prompt + f"Sentence: {input_sample}\n" + "Labels:"
    return output_prompt


def prompt(input_sample, examples):
    base_prompt = f"A word can be composed of one root and one or multiple affixed, \
                    segment the following sentence into its morphological constituents.\
                    The input will be a list of words in the sentence. \
                    The output format should be a list where each word from the input text is segmented and joined by a + sign.\
                    "

    return [
        {
            "role": "system",
            "content": "You are a linguist that helps in annotating data.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    results = []
    text = response["choices"][0]["message"]["content"]
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text
