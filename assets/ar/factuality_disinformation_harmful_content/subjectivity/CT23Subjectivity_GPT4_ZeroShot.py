from llmebench.datasets import CT23SubjectivityDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SubjectivityTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. Uses an prompt specified in Arabic.",
        "scores": {"Macro-F1": "0.725"},
    }


def config():
    return {
        "dataset": CT23SubjectivityDataset,
        "task": SubjectivityTask,
        "model": OpenAIModel,
        "general_args": {"test_split": "ar/dev"},
    }


def prompt(input_sample):
    prompt_string = (
        f"صنف الجملة إلى لاموضوعية أو موضوعية.\n\n"
        f"التغريدة: {input_sample}\n"
        f"التصنيف: \n"
    )
    return [
        {
            "role": "system",
            "content": "أنت خبير في تصنيف النصوص، ويمكنك تحليل المعلومات الموجودة في الجملة وتحديد ما إذا كانت الجملة موضوعية أم لاموضوعية.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()

    if "لاموضوعية" in label:
        label_fixed = "SUBJ"
    elif (
        label == "موضوعية" or label == "التصنيف: موضوعية" or "التصنيف: موضوعية" in label
    ):
        label_fixed = "OBJ"
    else:
        label_fixed = None

    return label_fixed
