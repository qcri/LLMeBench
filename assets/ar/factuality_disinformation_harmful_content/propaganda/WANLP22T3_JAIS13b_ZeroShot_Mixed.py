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


def prompt(input_sample):
    instruction = """
    "من القائمة، اختر التقنية الدعائية الأنسب للتغريدة: "no technique"، "Smears"، "Exaggeration/Minimisation"، "Loaded Language"، "Appeal to fear/prejudice"، "Name calling/Labeling"، "Slogans"، "Repetition"، "Doubt"،
    "Obfuscation, Intentional vagueness, Confusion"، "Flag-waving"، "Glittering generalities (Virtue)"، "Misrepresentation of Someone's Position (Straw Man)"، "Presenting Irrelevant Data (Red Herring)"، "Appeal to authority"، 
    "Whataboutism"، "Black-and-white Fallacy/Dictatorship"، "Thought-terminating cliché"، أو "Causal Oversimplification".
    """
    return [
        {
            "role": "user",
            "content": (
                f" \n{instruction}\n" + "التغريدة: " + input_sample + "التصنيف: "
            ),
        }
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()
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
    print("label: ", label)
    detected_labels = []

    if "بدون تقنية" in label:
        detected_labels.append(label_mapping["بدون تقنية"])
    if "تشويه" in label:
        detected_labels.append(label_mapping["تشويه"])
    if "تقليل" in label or "مبالغة" in label:
        detected_labels.append(label_mapping["مبالغة/تقليل"])
    if "لغة محملة" in label:
        detected_labels.append(label_mapping["لغة محملة"])
    if "التحيز" in label or "الخوف" in label:
        detected_labels.append(label_mapping["النداء إلى الخوف/التحيز"])
    if "الملصقات" in label or "التسمية" in label:
        detected_labels.append(label_mapping["التسمية/الملصقات"])
    if "الشعارات" in label:
        detected_labels.append(label_mapping["الشعارات"])
    if "التكرار" in label:
        detected_labels.append(label_mapping["التكرار"])
    if "الشك" in label:
        detected_labels.append(label_mapping["الشك"])
    if "الارتباك" in label or "الغموض المتعمد" in label or "التعمية" in label:
        detected_labels.append(label_mapping["التعمية/الغموض المتعمد/الارتباك"])
    if "التلويح بالعلم" in label:
        detected_labels.append(label_mapping["التلويح بالعلم"])
    if "التعميمات البراقة" in label or "الفضيلة" in label:
        detected_labels.append(label_mapping["التعميمات البراقة (الفضيلة)"])
    if "رجل القش" in label or "تحريف موقف شخص" in label:
        detected_labels.append(label_mapping["تحريف موقف شخص (رجل القش)"])
    if "السمكة الحمراء" in label or "عرض بيانات غير ذات صلة" in label:
        detected_labels.append(label_mapping["عرض بيانات غير ذات صلة (السمكة الحمراء)"])
    if "النداء إلى السلطة" in label:
        detected_labels.append(label_mapping["النداء إلى السلطة"])
    if "ماذا عن" in label:
        detected_labels.append(label_mapping["ماذا عن"])
    if "الأبيض والأسود" in label or "الديكتاتورية" in label:
        detected_labels.append(label_mapping["مغالطة الأبيض والأسود/الديكتاتورية"])
    if "القاطع للفكر" in label or "الكليشيه" in label:
        detected_labels.append(label_mapping["الكليشيه القاطع للفكر"])
    if "السببي" in label or "التبسيط" in label:
        detected_labels.append(label_mapping["التبسيط السببي"])

    return detected_labels
