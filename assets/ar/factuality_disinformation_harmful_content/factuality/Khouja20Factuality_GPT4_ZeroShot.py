import os

from llmebench.datasets import Khouja20FactualityDataset
from llmebench.models import GPTChatCompletionModel
from llmebench.tasks import Khouja20FactualityTask


def config():
    return {
        "dataset": Khouja20FactualityDataset,
        "dataset_args": {},
        "task": Khouja20FactualityTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/factuality_stance_khouja/claim/test.csv",
        },
    }


def prompt(input_sample):
    prompt_text = (
        "Detect whether the information in the sentence is factually true or false. "
        "Answer only by true or false.\n\n"
        + "Sentence: "
        + input_sample
        + "\nlabel: \n"
    )

    return [
        {
            "role": "system",
            "content": "You are an expert fact checker.",
        },
        {
            "role": "user",
            "content": prompt_text,
        },
    ]


def post_process(response):
    input_label = response["choices"][0]["message"]["content"]
    input_label = input_label.replace(".", "").strip().lower()

    if (
        "true" in input_label
        or "label: 1" in input_label
        or "label: yes" in input_label
    ):
        pred_label = "true"
    elif (
        "false" in input_label
        or "label: 0" in input_label
        or "label: no" in input_label
    ):
        pred_label = "false"
    else:
        print("label problem!! " + input_label)
        pred_label = None

    return pred_label
