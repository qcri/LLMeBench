import random
import re

from llmebench.datasets import WANLP22T3PropagandaDataset
from llmebench.models import FastChatModel
from llmebench.tasks import MultilabelPropagandaTask

random.seed(1333)


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "jais-13b-chat",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": WANLP22T3PropagandaDataset,
        "dataset_args": {"techniques_path": "classes.txt"},
        "task": MultilabelPropagandaTask,
        "model": FastChatModel,
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
            "max_tries": 3,
        },
    }


def few_shot_prompt(input_sample, examples):
    instruction = """
    "من القائمة، اختر التقنية الدعائية الأنسب للتغريدة: "بدون تقنية"، "تشويه"، "مبالغة/تقليل"، "لغة محملة"، "النداء إلى الخوف/التحيز"، "التسمية/الملصقات"، "الشعارات"، "التكرار"، "الشك"،
    "التعمية/الغموض المتعمد/الارتباك"، "التلويح بالعلم"، "التعميمات البراقة (الفضيلة)"، "تحريف موقف شخص (رجل القش)"، "عرض بيانات غير ذات صلة (السمكة الحمراء)"، "النداء إلى السلطة"، 
    "ماذا عن"، "مغالطة الأبيض والأسود/الديكتاتورية"، "الكليشيه القاطع للفكر"، أو "التبسيط السببي"."
    """
    label_mapping = {
        "no technique": "بدون تقنية",
        "Smears": "تشويه",
        "Exaggeration/Minimisation": "مبالغة/تقليل",
        "Loaded Language": "لغة محملة",
        "Appeal to fear/prejudice": "النداء إلى الخوف/التحيز",
        "Name calling/Labeling": "التسمية/الملصقات",
        "Slogans": "الشعارات",
        "Repetition": "التكرار",
        "Doubt": "الشك",
        "Obfuscation Intentional vagueness Confusion": (
            "التعمية/الغموض المتعمد/الارتباك"
        ),
        "Flag-waving": "التلويح بالعلم",
        "Glittering generalities (Virtue)": "التعميمات البراقة (الفضيلة)",
        "Misrepresentation of Someone's Position (Straw Man)": (
            "تحريف موقف شخص (رجل القش)"
        ),
        "Presenting Irrelevant Data (Red Herring)": (
            "عرض بيانات غير ذات صلة (السمكة الحمراء)"
        ),
        "Appeal to authority": "النداء إلى السلطة",
        "Whataboutism": "ماذا عن",
        "Black-and-white Fallacy/Dictatorship": "مغالطة الأبيض والأسود/الديكتاتورية",
        "Thought-terminating cliché": "الكليشيه القاطع للفكر",
        "Causal Oversimplification": "التبسيط السببي",
    }

    few_shot_text = instruction + "\n\nإليك بعض الأمثلة:\n\n"
    for index, example in enumerate(examples):
        print(f"Processing example {index + 1}")
        print(f"Example label: {example['label']}")

        try:
            labels = ", ".join(
                label_mapping[l] for l in example["label"] if example["label"]
            )
            print("Labels in few_shot:", labels)
        except KeyError as e:
            print(f"KeyError: {e} in example {index + 1}")
            labels = "Unknown Label"

    few_shot_text += f"الآن، قم بتقييم التغريدة الجديدة التالية:\nالتغريدة: '{input_sample}'\nالتصنيف: "
    return few_shot_text


def prompt(input_sample, examples):
    return [{"role": "user", "content": few_shot_prompt(input_sample, examples)}]


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()
    label = label.lower()
    label = re.sub(r"<[^>]+>", "", label)  # Remove any HTML-like tags
    label = label.lower()

    label_mapping = {
        "بدون تقنية": "no technique",
        "تشويه": "Smears",
        "مبالغة/تقليل": "Exaggeration/Minimisation",
        "لغة محملة": "Loaded Language",
        "النداء إلى الخوف/التحيز": "Appeal to fear/prejudice",
        "التسمية/الملصقات": "Name calling/Labeling",
        "الشعارات": "Slogans",
        "التكرار": "Repetition",
        "الشك": "Doubt",
        "التعمية/الغموض المتعمد/الارتباك": (
            "Obfuscation, Intentional vagueness, Confusion"
        ),
        "التلويح بالعلم": "Flag-waving",
        "التعميمات البراقة (الفضيلة)": "Glittering generalities (Virtue)",
        "تحريف موقف شخص (رجل القش)": (
            "Misrepresentation of Someone's Position (Straw Man)"
        ),
        "عرض بيانات غير ذات صلة (السمكة الحمراء)": (
            "Presenting Irrelevant Data (Red Herring)"
        ),
        "النداء إلى السلطة": "Appeal to authority",
        "ماذا عن": "Whataboutism",
        "مغالطة الأبيض والأسود/الديكتاتورية": "Black-and-white Fallacy/Dictatorship",
        "الكليشيه القاطع للفكر": "Thought-terminating cliché",
        "التبسيط السببي": "Causal Oversimplification",
    }

    detected_labels = []
    for arabic_label, english_label in label_mapping.items():
        if arabic_label in label:
            detected_labels.append(english_label)
        elif english_label.lower() in label:
            detected_labels.append(english_label)

    print("Detected labels:", detected_labels)

    # this is for duplicates values
    detected_labels = list(set(detected_labels))

    return detected_labels
