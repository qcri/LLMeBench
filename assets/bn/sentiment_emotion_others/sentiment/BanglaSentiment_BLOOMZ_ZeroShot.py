from llmebench.datasets import BanglaSentimentDataset
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
        "dataset": BanglaSentimentDataset,
        "dataset_args": {},
        "task": SentimentTask,
        "task_args": {},
        "model": PetalsModel,
        "model_args": {
            "class_labels": ["Positive", "Negative", "Neutral"],
            "max_tries": 10,
        },
    }


def prompt(input_sample):
    prompt_string = f"""Label the following text as Neutral Positive, or Negative. Provide only the label as your response. 

        text: {input_sample}

        label: """

    return {"prompt": prompt_string}


def post_process(response):
    label = response["outputs"].strip()
    label = label.replace("<s>", "").replace("</s>", "").strip()

    return label
