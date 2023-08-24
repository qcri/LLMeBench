import os 
import re 

from llmebench.datasets import ArabicPOSDataset
from llmebench.models import BLOOMPetalModel
from llmebench.tasks import ArabicPOSTask


def config():
    sets = [
        ("egy", "egy.pos/egy.data_5.test.src-trg.sent"),
        ("glf", "glf.pos/glf.data_5.test.src-trg.sent"),
        ("mgr", "mgr.pos/mgr.data_5.test.src-trg.sent"),
        ("lev", "lev.pos/lev.data_5.test.src-trg.sent"),
        ("msa", "WikiNewsTruth.txt.POS.tab"),
        ("XGLUE", "XGLUE/ar.test.src-trg.txt"),
    ]
    configs = []
    for name, testset in sets:
        configs.append(
            {
                "name": name,
                "config": {
                    "dataset": ArabicPOSDataset,
                    "dataset_args": {},
                    "task": ArabicPOSTask,
                    "task_args": {},
                    "model": BLOOMPetalModel,
                    "model_args": {
                        "api_url": os.environ["API_URL"],
                        "max_tries": 3,
                    },
                    "general_args": {
                        "data_path": "data/sequence_tagging_ner_pos_etc/POS/" + testset
                    },
                },
            }
        )
    return configs

def prompt(input_sample):
    tokens = input_sample.split()
    return {

        "prompt": f'Please provide the POS tags for each word in the input sentence. The input will be a list of words in the sentence. The output format should be a list of tuples, where each tuple consists of a word from the input text and its corresponding POS tag label from the tag label set: ["ABBREV", "ADJ", "ADV", "CASE", "CONJ", "DET", "EMOT", "FOREIGN", "FUT_PART", "HASH", "MENTION", "NEG_PART", "NOUN", "NSUFF", "NUM", "PART", "PREP", "PROG_PART", "PRON", "PUNC", "URL", "V"]. Note: Your response should include only a list of tuples, in the order that the words appear in the input sentence, with each tuple containing the corresponding POS tag label for a word.\nText:{tokens}\nPOS tags: '
    }

def post_process(response): 
    return response["outputs"]
