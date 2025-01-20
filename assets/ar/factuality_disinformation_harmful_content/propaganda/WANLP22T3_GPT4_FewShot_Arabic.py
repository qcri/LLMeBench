import ast
import random
import re

from llmebench.datasets import WANLP22T3PropagandaDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import MultilabelPropagandaTask


random.seed(1333)


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "GPT-4o-2024-05-22",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": WANLP22T3PropagandaDataset,
        "dataset_args": {"techniques_path": "classes.txt"},
        "task": MultilabelPropagandaTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": [
                "no technique",
                "Smears",
                "Exaggeration/Minimisation",
                "Loaded Language",
                "Appeal to fear/prejudice",
                "Name calling/Labeling",
                "Slogans",
                "Repetition",
                "Doubt",
                "Obfuscation, Intentional vagueness, Confusion",
                "Flag-waving",
                "Glittering generalities (Virtue)",
                "Misrepresentation of Someone's Position (Straw Man)",
                "Presenting Irrelevant Data (Red Herring)",
                "Appeal to authority",
                "Whataboutism",
                "Black-and-white Fallacy/Dictatorship",
                "Thought-terminating cliché",
                "Causal Oversimplification",
            ],
            "max_tries": 30,
        },
    }


def translate_labels(label):
    label_mapping = {
        "no technique": "بدون تقنية",
        "Smears": "تشويه",
        "Exaggeration/Minimisation": "مبالغة/تقليل",
        "Loaded Language": "لغة محملة بالمشاعر",
        "Appeal to fear/prejudice": "الاحتكام إلى الخوف/التحيز",
        "Name calling/Labeling": "التسمية/الملصقات",
        "Slogans": "الشعارات",
        "Repetition": "التكرار",
        "Doubt": "الشك",
        "Obfuscation, Intentional vagueness, Confusion": "التعمية/الغموض المتعمد/الارتباك",
        "Flag-waving": "التلويح بالعلم",
        "Glittering generalities (Virtue)": "التعميمات البراقة (الفضيلة)",
        "Misrepresentation of Someone's Position (Straw Man)": "تحريف موقف شخص (مغالطة رجل القش)",
        "Presenting Irrelevant Data (Red Herring)": "عرض بيانات غير ذات صلة (السمكة الحمراء)",
        "Appeal to authority": "الاحتكام إلى السلطة",
        "Whataboutism": "ماذا عن",
        "Black-and-white Fallacy/Dictatorship": "مغالطة الأبيض والأسود/الديكتاتورية",
        "Thought-terminating cliché": "الكليشيه القاطع للفكر",
        "Causal Oversimplification": "التبسيط السببي",
    }
    return label_mapping.get(label, label)


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\nاليك بعض الأمثلة:\n\n"
    for index, example in enumerate(examples):
        tech_str = ", ".join([f"'{translate_labels(t)}'" for t in example["label"]])
        out_prompt += (
            f"مثال {index}:\nالتغريدة: {example['input']}\nالتصنيف: {tech_str}\n\n"
        )
    out_prompt += f"التغريدة: {input_sample}\nالتصنيف: \n"
    return out_prompt


def prompt(input_sample, examples):
    base_prompt = """
        "من القائمة، اختر التقنية الدعائية الأنسب للتغريدة: "بدون تقنية"، "تشويه"، "مبالغة/تقليل"، "لغة محملة بالمشاعر"، "الاحتكام إلى الخوف/التحيز"، "التسمية/الملصقات"، "الشعارات"، "التكرار"، "الشك"،
        "التعمية/الغموض المتعمد/الارتباك"، "التلويح بالعلم"، "التعميمات البراقة (الفضيلة)"، "تحريف موقف شخص (مغالطة رجل القش)"، "عرض بيانات غير ذات صلة (السمكة الحمراء)"، "الاحتكام إلى السلطة"، 
        "ماذا عن"، "مغالطة الأبيض والأسود/الديكتاتورية"، "الكليشيه القاطع للفكر"، أو "التبسيط السببي"."
        """

    return [
        {
            "role": "system",
            "content": "أنت خبير في تحليل محتوى وسائل التواصل الاجتماعي.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]  # .lower()
    label = label.strip().lower()
    if (
        "لا يوجد في النص" in label
        or label == "'no technique'"
        or "doesn't" in label
        or "does not" in label
        or "لا يحتوي" in label
        or "لا يستخدم" in label
    ):
        return []
    label_mapping = {
        "بدون تقنية": "no technique",
        "تشويه": "Smears",
        "مبالغة/تقليل": "Exaggeration/Minimisation",
        "لغة محملة بالمشاعر": "Loaded Language",
        "الاحتكام إلى الخوف/التحيز": "Appeal to fear/prejudice",
        "التسمية/الملصقات": "Name calling/Labeling",
        "الشعارات": "Slogans",
        "التكرار": "Repetition",
        "الشك": "Doubt",
        "التعمية/الغموض المتعمد/الارتباك": "Obfuscation, Intentional vagueness, Confusion",
        "التلويح بالعلم": "Flag-waving",
        "التعميمات البراقة (الفضيلة)": "Glittering generalities (Virtue)",
        "تحريف موقف شخص (مغالطة رجل القش)": "Misrepresentation of Someone's Position (Straw Man)",
        "عرض بيانات غير ذات صلة (السمكة الحمراء)": "Presenting Irrelevant Data (Red Herring)",
        "الاحتكام إلى السلطة": "Appeal to authority",
        "ماذا عن": "Whataboutism",
        "مغالطة الأبيض والأسود/الديكتاتورية": "Black-and-white Fallacy/Dictatorship",
        "الكليشيه القاطع للفكر": "Thought-terminating cliché",
        "التبسيط السببي": "Causal Oversimplification",
    }

    detected_labels = [
        english_label
        for arabic_label, english_label in label_mapping.items()
        if arabic_label in label or english_label.lower() in label
    ]

    final_labels = [l for l in detected_labels if "no technique" not in l]

    return list(set(final_labels))
