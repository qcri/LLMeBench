from llmebench.datasets import AdultDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import AdultTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Macro-F1": "0.832"},
    }


def config():
    return {
        "dataset": AdultDataset,
        "task": AdultTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["ADULT", "NOT_ADULT"],
            "max_tries": 30,
        },
        "general_args": {
            "fewshot": {
                "deduplicate": True,
            },
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n\n"
    out_prompt = out_prompt + "اليك بعض الامثلة:\n\n"
    for index, example in enumerate(examples):
        label = "للبالغين" if example["label"] == "ADULT" else "ليست للبالغين"

        out_prompt = (
            out_prompt
            + "مثال "
            + str(index)
            + ":"
            + "\n"
            + "التغريدة: "
            + example["input"]
            + "\التصنيف: "
            + label
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "التغريدة: " + input_sample + "\التصنيف: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = "هل تحتوي التغريدة التالية على لغة للبالغين؟ اجب 'للبالغين' أو 'ليست للبالغين' بدون أي تفسير."
    return [
        {
            "role": "system",
            "content": "أنت محلل خبير، يمكنك تحديد وتصنيف المحتوى الخاص بالبالغين في التغريدة.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    if not response:
        return None

    label = response["choices"][0]["message"]["content"]
    label = label.replace("التصنيف:", "").strip()
    label = label.replace("label:", "").strip()
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
