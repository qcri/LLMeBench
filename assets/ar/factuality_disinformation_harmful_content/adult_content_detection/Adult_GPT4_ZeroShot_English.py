from llmebench.datasets import AdultDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import AdultTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Macro-F1": "0.727"},
    }


def config():
    return {
        "dataset": AdultDataset,
        "task": AdultTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["ADULT", "NOT_ADULT"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    prompt_string = (
        # f'Classify whether the following Arabic sentence is appropriate for children. If not appropriate, just write "ADULT" without explaination, and if appropriate, just write "NOT_ADULT" without explaination \n {input_sample}'
        f'Does the following tweet contain adult content? label it as "ADULT" if it does, or "NOT_ADULT" otherwise. Return label only.\n\n'
        f"tweet: {input_sample}\n"
        f"label: \n"
    )
    return [
        {
            "role": "system",
            "content": "You are an expert annotator, you can identify and label adult content within a tweet.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]
    label = label.replace("التصنيف:", "").strip()
    label = label.replace("label:", "").strip()

    label = label.replace("<s>", "").replace("</s>", "")
    label = label.lower()

    if (
        "ليس" in label
        or "ليست" in label
        or "not" in label
        or "no" in label
        or "غير" in label
        or "لا" in label
        or "not_adult" in label
        or "not adult" in label
    ):
        return "NOT_ADULT"
    elif "للبالغين" in label or "نعم" in label or "adult" in label:
        return "ADULT"
    else:
        return None
