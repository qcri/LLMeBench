import os
import re

from llmebench.datasets import ANERcorpDataset
from llmebench.models import HuggingFaceInferenceAPIModel, HuggingFaceTaskTypes
from llmebench.tasks import NERTask


def config():
    return {
        "dataset": ANERcorpDataset,
        "dataset_args": {},
        "task": NERTask,
        "task_args": {},
        "model": HuggingFaceInferenceAPIModel,
        "model_args": {
            "task_type": HuggingFaceTaskTypes.Named_Entity_Recognition,
            "inference_api_url": "https://api-inference.huggingface.co/models/CAMeL-Lab/bert-base-arabic-camelbert-ca-ner",
            "api_token": os.environ["HUGGINGFACE_API_TOKEN"],
            "max_tries": 5,
        },
        "general_args": {
            "data_path": "data/sequence_tagging_ner_pos_etc/NER/AnerCorp/ANERCorp_CamelLab_test.txt"
        },
    }


def prompt(input_sample):
    return {"inputs": input_sample}


def post_process(response):
    # response = response["choices"][0]["text"]
    import pdb

    pdb.set_trace()
    possible_tags = [
        "B-PERS",
        "I-PERS",
        "B-LOC",
        "I-LOC",
        "B-ORG",
        "I-ORG",
        "O",
        "B-MISC",
        "I-MISC",
    ]
    mapping = {
        "PER-B": "B-PERS",
        "PER-I": "I-PERS",
        "ORG-B": "B-ORG",
        "ORG-I": "I-ORG",
        "LOC-B": "B-LOC",
        "LOC-I": "I-LOC",
        "MISC-B": "B-MISC",
        "MISC-I": "I-MISC",
    }

    return response

    starts = [(r["start"], r) for r in response]
    starts.sort(key=lambda x: x[0], reverse=True)

    for r in starts:
        r[1]["end"]

    matches = re.findall(r"\((.*?)\)", response)
    if matches:
        cleaned_response = []
        for match in matches:
            elements = match.split(",")
            try:
                cleaned_response.append(elements[1])
            except:
                cleaned_response.append("O")

        cleaned_response = [
            sample.replace("'", "").strip() for sample in cleaned_response
        ]
        final_cleaned_response = []
        for elem in cleaned_response:
            if elem in possible_tags:
                final_cleaned_response.append(elem)
            elif elem in mapping:
                final_cleaned_response.append(mapping[elem])
            else:
                final_cleaned_response.append("O")
    else:
        final_cleaned_response = None
    return final_cleaned_response
