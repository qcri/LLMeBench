from llmebench.datasets import ANSStanceDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import StanceTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Macro-F1": "0.721"},
    }


def config():
    return {
        "dataset": ANSStanceDataset,
        "dataset_args": {},
        "task": StanceTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["agree", "disagree"],
            "max_tries": 30,
        },
    }


def prompt(input_sample, examples):
    prompt_string = "Given a reference sentence and a claim, predict whether the claim agrees or disagrees with the reference sentence. Reply only using 'agree', 'disagree', or use 'other' if the sentence and claim are unrelated.\n\n"

    return [
        {
            "role": "system",
            "content": "You are a fact checking expert.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, prompt_string, examples),
        },
    ]


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt
    for example in examples:
        ref_s = example["input"]["sentence_1"]
        claim = example["input"]["sentence_2"]

        out_prompt = (
            out_prompt
            + "reference sentence: "
            + ref_s
            + "\nclaim: "
            + claim
            + "\nlabel: "
            + example["label"]
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the label blank

    ref_s = input_sample["sentence_1"]
    claim = input_sample["sentence_2"]

    out_prompt = (
        out_prompt
        + "reference sentence: "
        + ref_s
        + "\nclaim: "
        + claim
        + "\nlabel: \n"
    )

    # print("=========== FS Prompt =============\n")
    # print(out_prompt)

    return out_prompt


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()
    label = label.replace("label:", "").strip()

    label_fixed = None

    if "unrelated" in label or "other" in label:
        label_fixed = "other"
    elif "disagree" in label:
        label_fixed = "disagree"
    elif label == "agree":
        label_fixed = "agree"

    return label_fixed
