import os
import random
import re

from arabic_llm_benchmark.datasets import SubjectivityDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import SubjectivityTask


random.seed(1333)


def config():
    return {
        "dataset": SubjectivityDataset,
        "dataset_args": {},
        "task": SubjectivityTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["SUBJ", "OBJ"],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/subjectivity/dev_ar.tsv"
        },
    }


def prompt(input_sample):
    prompt_string = (
        f'صنف الجملة إلى لاموضوعية أو موضوعية.\n\n'
        f"التغريدة: {input_sample}\n"
        f"التصنيف: \n"
    )
    return [
        {
            "role": "system",
            "content": "أنت خبير في تصنيف النصوص، ويمكنك تحليل المعلومات الموجودة في الجملة وتحديد ما إذا كانت الجملة موضوعية أم لاموضوعية."
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
    elif label == "موضوعية" or label == "التصنيف: موضوعية" or "التصنيف: موضوعية" in label:
        label_fixed = "OBJ"
    else:
        label_fixed = None

    return label_fixed
