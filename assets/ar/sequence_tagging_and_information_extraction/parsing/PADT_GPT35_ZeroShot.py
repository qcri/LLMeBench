import re

from llmebench.datasets import PADTDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import ArabicParsingTask


def config():
    return {
        "dataset": PADTDataset,
        "dataset_args": {},
        "task": ArabicParsingTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sequence_tagging_ner_pos_etc/Parsing/arabic_PADT_test_gs.conll"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are a linguist that helps in annotating data.",
        "messages": [
            {
                "sender": "user",
                "text": f"Given the following features (in order: ID, Form, Lemma, CPostTag, POSTag, Features), predict the Head of each token in the following sentence, which is either a value of a related ID or 0. A value of zero means the token attaches to the virtual root node: {input_sample}",
            }
        ],
    }


def post_process(response):
    output = response["choices"][0]["text"]
    results = {}
    if len(output):
        output = output.strip().split("\n")

        for o in output:
            src, tgt = (re.sub(r"[^0-9]+", "\t", o)).split("\t")
            results[src] = tgt
        output = results
    else:
        output = None
    return output
