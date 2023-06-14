import os

from arabic_llm_benchmark.datasets import XNLIDataset
from arabic_llm_benchmark.models import GPTModel
from arabic_llm_benchmark.tasks import XNLITask


def config():
    return {
        "dataset": XNLIDataset,
        "dataset_args": {},
        "task": XNLITask,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "gpt",
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/XNLI/xnli.test.ar.tsv"
        },
    }


def prompt(input_sample):
    sent1,sent2 = input_sample.split('\t')

    prompt_text = "You are provided with a premise and a hypothesis. Your task is to classify the hypothesis as either true (entailment), false (contradiction), or unknown (neutral) based on the given premise. The output should only be exactly one of three labels: true, false or unknown."
    prompt_text = prompt_text + "\nPremise: " + sent1 + \
                  "\nHypothesis: " + sent2 + "\n"
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": prompt_text,
            }
        ],
    }


def post_process(response):
    input_label = response["choices"][0]["text"]
    input_label = input_label.replace(".", "").strip().lower()

    if ("neutral" in input_label or "unknown" in input_label):
        pred_label = "neutral"
    elif ("true" in input_label or "entailment" in input_label):
        pred_label = "entailment"
    elif ("false" in input_label or "contradiction" in input_label):
        pred_label = "contradiction"
    else:
        print(input_label)
        pred_label = None

    return pred_label
