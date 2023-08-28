import os
import random

from arabic_llm_benchmark.datasets import NewsCatASNDDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import NewsCatASNDTask


random.seed(1333)


def config():
    return {
        "dataset": NewsCatASNDDataset,
        "dataset_args": {},
        "task": NewsCatASNDTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": [
                "crime-war-conflict",
                "spiritual",
                "health",
                "politics",
                "human-rights-press-freedom",
                "education",
                "business-and-economy",
                "art-and-entertainment",
                "others",
                "science-and-technology",
                "sports",
                "environment",
            ],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/news_categorization/Arabic_Social_Media_News_Dataset_ASND/sm_news_ar_tst.csv"
        },
    }


def prompt(input_sample):
    prompt_string = (
        f"صنف التغريدة التالية إلى واحدة من الفئات التالية: "
        f"جريمة-حرب-صراع ، روحي ، صحة ، سياسة ، حقوق-الإنسان-حرية-الصحافة ، "
        f"تعليم ، أعمال-و-اقتصاد ، فن-و-ترفيه ، أخرى ، "
        f"علم-و-تكنولوجيا ، رياضة ، بيئة\n"
        f"\nالتغريدة: {input_sample}"
        f"\nالفئة: \n"
    )

    return [
        {
            "role": "system",
            "content": "أنت خبير في تصنيف التغريدات وتعرف كيف تصنف تغريدات الأخبار.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    if "جريمة-حرب-صراع" in label or "صراع-حرب" in label:
        label_fixed = "crime-war-conflict"
    elif "روحي" in label:
        label_fixed = "spiritual"
    elif "صحة" in label:
        label_fixed = "health"
    elif "سياسة" in label:
        label_fixed = "politics"
    elif "حقوق-الإنسان-حرية-الصحافة" in label:
        label_fixed = "human-rights-press-freedom"
    elif "تعليم" in label:
        label_fixed = "education"
    elif "أعمال-و-اقتصاد" in label:
        label_fixed = "business-and-economy"
    elif "فن-و-ترفيه" in label or "ترفيه" in label:
        label_fixed = "art-and-entertainment"
    elif "أخرى" in label:
        label_fixed = "others"
    elif "علم-و-تكنولوجيا" in label:
        label_fixed = "science-and-technology"
    elif "رياضة" in label:
        label_fixed = "sports"
    elif "بيئة" in label:
        label_fixed = "environment"
    else:
        label_fixed = "others"

    return label_fixed
