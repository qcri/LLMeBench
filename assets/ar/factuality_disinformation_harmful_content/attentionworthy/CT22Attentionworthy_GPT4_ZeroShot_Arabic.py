from llmebench.datasets import CT22AttentionworthyDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import AttentionworthyTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Weighted-F1": "0.257"},
    }


def config():
    return {
        "dataset": CT22AttentionworthyDataset,
        "task": AttentionworthyTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": [
                "yes_discusses_action_taken",
                "harmful",
                "yes_discusses_cure",
                "yes_asks_question",
                "no_not_interesting",
                "yes_other",
                "yes_blame_authorities",
                "yes_contains_advice",
                "yes_calls_for_action",
            ],
            "max_tries": 30,
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    prompt_string = (
        f'هل تتطلب "التغريدة" انتباه الصحفيين، ومدققي الحقائق، والجهات الحكومية؟ قم بتصنيف "التغريدة" التالية إلى واحدة من الفئات التالية: تناقش_الإجراء_المتخذ، ضارة، تناقش_العلاج، تطرح_سؤال، غير_مثيرة_للاهتمام، آخر، تلوم_السلطات، تحتوي_على_نصيحة، تدعو_لإجراء. قدم التصنيف فقط.\n\n'
        f"التغريدة: {input_sample}\n"
        f"التصنيف: \n"
    )
    return [
        {
            "role": "system",
            "content": "أنت خبير في وسائل التواصل الاجتماعي. يمكنك تحليل و تصنيف التغريدات.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label = (
        label.replace(" - ", ", ")
        .replace(",", "")
        .replace(".", "")
        .replace("label:", "")
        .replace("التصنيف: ", "")
    )
    label_fixed = label.lower().strip()  # تحويل إلى أحرف صغيرة وإزالة الفراغات الزائدة

    if label_fixed.startswith("لا"):
        label_fixed = "no_not_interesting"
    elif "تناقش_العلاج" in label_fixed:
        label_fixed = "yes_discusses_cure"
    elif "ضارة" in label_fixed:
        label_fixed = "harmful"
    elif "تلوم_السلطات" in label_fixed:
        label_fixed = "yes_blame_authoritie"
    elif "تدعو_لإجراء" in label_fixed:
        label_fixed = "yes_calls_for_action"
    elif "تناقش_الإجراء_المتخذ" in label_fixed:
        label_fixed = "yes_discusses_action_taken"
    elif "علاج" in label_fixed:
        label_fixed = "yes_discusses_cure"
    elif "تطرح سؤال" in label_fixed:
        label_fixed = "yes_asks_question"
    elif "تطرح_سؤال" in label_fixed:
        label_fixed = "yes_asks_question"
    elif "غير_مثيرة_للاهتمام" in label_fixed:
        label_fixed = "no_not_interesting"
    elif "آخر" in label_fixed:
        label_fixed = "yes_other"
    elif "السلطات" in label_fixed:
        label_fixed = "yes_blame_authorities"
    elif "نصيحة" in label_fixed:
        label_fixed = "yes_contains_advice"
    elif "تدعو لإجراء" in label_fixed:
        label_fixed = "yes_calls_for_action"
    else:
        label_fixed = None

    return label_fixed
