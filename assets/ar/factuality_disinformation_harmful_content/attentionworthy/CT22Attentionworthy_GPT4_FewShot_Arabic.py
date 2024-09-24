from llmebench.datasets import CT22AttentionworthyDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import AttentionworthyTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Weighted-F1": "0.412"},
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
        "general_args": {"test_split": "ar", "fewshot": {"train_split": "ar"}},
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    label_translation = {
        "yes_discusses_action_taken": "تناقش_الإجراء_المتخذ",
        "harmful": "ضارة",
        "yes_discusses_cure": "تناقش_العلاج",
        "yes_asks_question": "تطرح_سؤال",
        "no_not_interesting": "غير_مثيرة_للاهتمام",
        "yes_other": "آخر",
        "yes_blame_authorities": "تلوم_السلطات",
        "yes_contains_advice": "تحتوي_على_نصيحة",
        "yes_calls_for_action": "تدعو_لإجراء",
    }

    out_prompt = base_prompt + "\n"
    for example in examples:
        translated_label = label_translation.get(example["label"], example["label"])
        out_prompt = (
            out_prompt
            + "التغريدة: "
            + example["input"]
            + "\nالتصنيف: "
            + translated_label
            + "\n\n"
        )
    out_prompt = out_prompt + "التغريدة: " + input_sample + "\التصنيف: \n"
    return out_prompt


def prompt(input_sample, examples):
    base_prompt = f'هل تتطلب "التغريدة" انتباه الصحفيين، ومدققي الحقائق، والجهات الحكومية؟ قم بتصنيف "التغريدة" التالية إلى واحدة من الفئات التالية: تناقش_الإجراء_المتخذ، ضارة، تناقش_العلاج، تطرح_سؤال، غير_مثيرة_للاهتمام، آخر، تلوم_السلطات، تحتوي_على_نصيحة، تدعو_لإجراء. قدم التصنيف فقط.\n\n'

    return [
        {
            "role": "system",
            "content": "أنت خبير في وسائل التواصل الاجتماعي. يمكنك تحليل و تصنيف التغريدات.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
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
