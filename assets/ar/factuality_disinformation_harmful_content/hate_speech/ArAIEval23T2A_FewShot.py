import re

from llmebench.datasets import ArAIEVAL232A
from llmebench.models import OpenAIModel
from llmebench.tasks import ClassificationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
    }


def config():
    return {
        "dataset": ArAIEVAL232A,
        "task": ClassificationTask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
        "general_args": {"data_path": "data/task1A_test.jsonl"},
    }


def prompt(input_sample, examples):
    prompt_text = (
        f"Disinformation is defined as fabricated or deliberately manipulated text, speech or visual context, and also intentionally created conspiracy theories or rumors. It can contain hate speech, offensive, spam and harmful content.\n\n"
        f"Illustrative examples are provided below:\n\n"
    )

    fs_prompt = few_shot_prompt(input_sample, prompt_text, examples)
    return [
        {
            "role": "system",
            "content": "You are an expert social media analyst.",
        },
        {
            "role": "user",
            "content": fs_prompt,
        },
    ]


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt
    for example in examples:
        sent = example["input"]
        label = example["label"]

        out_prompt = out_prompt + "text: " + sent + "\n" + "label: " + label + "\n\n"

    out_prompt = (
        out_prompt
        + "Read the text below and decide whether it contain such content. If so, answer only as disinfo or no-disinfo\n\n"
    )
    out_prompt = out_prompt + "text: " + input_sample + "\nlabel: \n"

    # print("=========== FS Prompt =============\n")
    # print(out_prompt)

    return out_prompt


def post_process(response):
    input_label = response["choices"][0]["message"]["content"]
    pred_label = input_label.replace(".", "").strip().lower()

    pred_label = pred_label.replace("label:", "").strip()

    if "no-disinfo" == pred_label or "disinfo" == pred_label:
        return pred_label
    else:
        pred_label = None

    return pred_label
