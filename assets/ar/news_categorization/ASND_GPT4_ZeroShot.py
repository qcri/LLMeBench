from llmebench.datasets import ASNDDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import NewsCategorizationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. Uses an prompt specified in Arabic.",
        "scores": {"Macro-F1": "0.739"},
    }


def config():
    return {
        "dataset": ASNDDataset,
        "task": NewsCategorizationTask,
        "model": OpenAIModel,
    }


def prompt(input_sample):
    prompt_string = (
        f"صنف التغريدة التالية إلى واحدة من الفئات التالية: "
        f"جريمة-حرب-صراع ، روحي-ديني ، صحة ، سياسة ، حقوق-الإنسان-حرية-الصحافة ، "
        f"تعليم ، أعمال-اقتصاد ، فن-ترفيه ، أخرى ، "
        f"علوم-تكنولوجيا ، رياضة ، بيئة\n"
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
    elif "روحي" in label or "ديني" in label:
        label_fixed = "spiritual"
    elif "صحة" in label:
        label_fixed = "health"
    elif "سياسة" in label:
        label_fixed = "politics"
    elif "حقوق-الإنسان-حرية-الصحافة" in label:
        label_fixed = "human-rights-press-freedom"
    elif "تعليم" in label:
        label_fixed = "education"
    elif "أعمال-و-اقتصاد" in label or "أعمال" in label or "اقتصاد" in label:
        label_fixed = "business-and-economy"
    elif "فن-و-ترفيه" in label or "ترفيه" in label:
        label_fixed = "art-and-entertainment"
    elif "أخرى" in label:
        label_fixed = "others"
    elif "علم-و-تكنولوجيا" in label or "علوم" in label or "تكنولوجيا" in label:
        label_fixed = "science-and-technology"
    elif "رياضة" in label:
        label_fixed = "sports"
    elif "بيئة" in label:
        label_fixed = "environment"
    else:
        label_fixed = "others"

    return label_fixed
