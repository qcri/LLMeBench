from llmebench.datasets import BanglaNewsCategorizationDataset
from llmebench.models import PetalsModel
from llmebench.tasks import SentimentTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
    }


def config():
    return {
        "dataset": BanglaNewsCategorizationDataset,
        "task": SentimentTask,
        "model": PetalsModel,
        "model_args": {
            "class_labels": ['entertainment', 'state', 'sports', 'national', 'kolkata', 'international'],
            "max_tries": 10,
        },
    }


def prompt(input_sample):
    prompt_string = f"""Label the following news as 'entertainment', 'state', 'sports', 'national', 'kolkata', or 'international'. Provide only the label as your response. 

        news: {input_sample}

        label: """

    return {"prompt": prompt_string}


def post_process(response):
    label = response["outputs"].strip()
    label = label.replace("<s>", "").replace("</s>", "").strip()

    return label
