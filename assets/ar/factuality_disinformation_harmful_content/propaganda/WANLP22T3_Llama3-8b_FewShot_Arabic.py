# import random
# import re

# from llmebench.datasets import WANLP22T3PropagandaDataset
# from llmebench.models import AzureModel
# from llmebench.tasks import MultilabelPropagandaTask

# random.seed(1333)

# def metadata():
#     return {
#         "author": "Arabic Language Technologies, QCRI, HBKU",
#         "model": "LLama 3 8b",
#         "description": "Deployed on Azure.",
#         "scores": {"Micro-F1": "0.52864"}, # 1 - 0.3631 , 3- 0.48027 , 5- 0.52864
#     }

# def config():
#     return {
#         "dataset": WANLP22T3PropagandaDataset,
#         "dataset_args": {"techniques_path": "classes.txt"},
#         "task": MultilabelPropagandaTask,
#         "model": AzureModel,
#         "model_args": {
#             "class_labels": [
#                 "no technique",
#                 "Smears",
#                 "Exaggeration/Minimisation",
#                 "Loaded Language",
#                 "Appeal to fear/prejudice",
#                 "Name calling/Labeling",
#                 "Slogans",
#                 "Repetition",
#                 "Doubt",
#                 "Obfuscation, Intentional vagueness, Confusion",
#                 "Flag-waving",
#                 "Glittering generalities (Virtue)",
#                 "Misrepresentation of Someone's Position (Straw Man)",
#                 "Presenting Irrelevant Data (Red Herring)",
#                 "Appeal to authority",
#                 "Whataboutism",
#                 "Black-and-white Fallacy/Dictatorship",
#                 "Thought-terminating cliché",
#                 "Causal Oversimplification",
#             ],
#             "max_tries": 3,
#         },
#     }

# def few_shot_prompt(input_sample, examples):
#     instruction = """
#         "من القائمة، اختر التقنية الدعائية الأنسب للتغريدة: "بدون تقنية"، "تشويه"، "مبالغة/تقليل"، "لغة محملة"، "النداء إلى الخوف/التحيز"، "التسمية/الملصقات"، "الشعارات"، "التكرار"، "الشك"،
#         "التعمية/الغموض المتعمد/الارتباك"، "التلويح بالعلم"، "التعميمات البراقة (الفضيلة)"، "تحريف موقف شخص (رجل القش)"، "عرض بيانات غير ذات صلة (السمكة الحمراء)"، "النداء إلى السلطة"،
#         "ماذا عن"، "مغالطة الأبيض والأسود/الديكتاتورية"، "الكليشيه القاطع للفكر"، أو "التبسيط السببي"."
#         """
#     label_mapping = {
#         "no technique": "بدون تقنية",
#         "Smears": "تشويه",
#         "Exaggeration/Minimisation": "مبالغة/تقليل",
#         "Loaded Language": "لغة محملة",
#         "Appeal to fear/prejudice": "النداء إلى الخوف/التحيز",
#         "Name calling/Labeling": "التسمية/الملصقات",
#         "Slogans": "الشعارات",
#         "Repetition": "التكرار",
#         "Doubt": "الشك",
#         "Obfuscation Intentional vagueness Confusion": "التعمية/الغموض المتعمد/الارتباك",
#         "Flag-waving": "التلويح بالعلم",
#         "Glittering generalities (Virtue)": "التعميمات البراقة (الفضيلة)",
#         "Misrepresentation of Someone's Position (Straw Man)": "تحريف موقف شخص (رجل القش)",
#         "Presenting Irrelevant Data (Red Herring)": "عرض بيانات غير ذات صلة (السمكة الحمراء)",
#         "Appeal to authority": "النداء إلى السلطة",
#         "Whataboutism": "ماذا عن",
#         "Black-and-white Fallacy/Dictatorship": "مغالطة الأبيض والأسود/الديكتاتورية",
#         "Thought-terminating cliché": "الكليشيه القاطع للفكر",
#         "Causal Oversimplification": "التبسيط السببي"
#     }

#     few_shot_text = instruction + "\n\nإليك بعض الأمثلة:\n\n"
#     for index, example in enumerate(examples):
#         labels_list = [label_mapping.get(label,"") for label in example["label"]]
#         labels = ", ".join(labels_list)
#         few_shot_text += (
#             f"مثال {index + 1}:\n"
#             f"التغريدة: '{example['input']}'\n"
#             f"التصنيف: {labels}\n\n"
#         )

#     few_shot_text += f"الآن، قم بتقييم التغريدة الجديدة التالية:\nالتغريدة: '{input_sample}'\nالتصنيف: "
#     return few_shot_text

# def few_shot_prompt(input_sample, base_prompt, examples):
#     label_mapping = {
#         "no technique": "بدون تقنية",
#         "Smears": "تشويه",
#         "Exaggeration/Minimisation": "مبالغة/تقليل",
#         "Loaded Language": "لغة محملة",
#         "Appeal to fear/prejudice": "النداء إلى الخوف/التحيز",
#         "Name calling/Labeling": "التسمية/الملصقات",
#         "Slogans": "الشعارات",
#         "Repetition": "التكرار",
#         "Doubt": "الشك",
#         "Obfuscation Intentional vagueness Confusion": "التعمية/الغموض المتعمد/الارتباك",
#         "Flag-waving": "التلويح بالعلم",
#         "Glittering generalities (Virtue)": "التعميمات البراقة (الفضيلة)",
#         "Misrepresentation of Someone's Position (Straw Man)": "تحريف موقف شخص (رجل القش)",
#         "Presenting Irrelevant Data (Red Herring)": "عرض بيانات غير ذات صلة (السمكة الحمراء)",
#         "Appeal to authority": "النداء إلى السلطة",
#         "Whataboutism": "ماذا عن",
#         "Black-and-white Fallacy/Dictatorship": "مغالطة الأبيض والأسود/الديكتاتورية",
#         "Thought-terminating cliché": "الكليشيه القاطع للفكر",
#         "Causal Oversimplification": "التبسيط السببي"
#     }

#     out_prompt = base_prompt + "\n"
#     out_prompt = out_prompt + "اليك بعض الأمثلة:\n\n"
#     for index, example in enumerate(examples):
#         tech_str = ""
#         for t in example["label"]:
#             tech_str += "'" + label_mapping[t] + "', "

#         out_prompt = (
#             out_prompt
#             + "مثال "
#             + str(index)
#             + ":"
#             + "\n"
#             + "التغريدة: "
#             + example["input"]
#             + "\التصنيف: "
#             + tech_str
#             + "\n\n"
#         )

#     # Append the sentence we want the model to predict for but leave the Label blank
#     out_prompt = out_prompt + "التغريدة: " + input_sample + "\التصنيف: \n"

#     return out_prompt

# def prompt(input_sample, examples):
#     return [
#         {
#             "role": "user",
#             "content": few_shot_prompt(input_sample, examples)
#         }
#     ]

# def post_process(response):
#     if not response or 'error' in response or 'output' not in response:
#         print("Error or missing output in response:", response)
#         return None

#     label = response["output"].strip().lower()
#     label = re.sub(r'<[^>]+>', '', label)  # Remove any HTML-like tags
#     label = label.lower()

#     label_mapping = {
#         "بدون تقنية": "no technique",
#         "تشويه": "Smears",
#         "مبالغة/تقليل": "Exaggeration/Minimisation",
#         "لغة محملة": "Loaded Language",
#         "النداء إلى الخوف/التحيز": "Appeal to fear/prejudice",
#         "التسمية/الملصقات": "Name calling/Labeling",
#         "الشعارات": "Slogans",
#         "التكرار": "Repetition",
#         "الشك": "Doubt",
#         "التعمية/الغموض المتعمد/الارتباك": "Obfuscation, Intentional vagueness, Confusion",
#         "التلويح بالعلم": "Flag-waving",
#         "التعميمات البراقة (الفضيلة)": "Glittering generalities (Virtue)",
#         "تحريف موقف شخص (رجل القش)": "Misrepresentation of Someone's Position (Straw Man)",
#         "عرض بيانات غير ذات صلة (السمكة الحمراء)": "Presenting Irrelevant Data (Red Herring)",
#         "النداء إلى السلطة": "Appeal to authority",
#         "ماذا عن": "Whataboutism",
#         "مغالطة الأبيض والأسود/الديكتاتورية": "Black-and-white Fallacy/Dictatorship",
#         "الكليشيه القاطع للفكر": "Thought-terminating cliché",
#         "التبسيط السببي": "Causal Oversimplification"
#     }

#     detected_labels = []
#     for arabic_label, english_label in label_mapping.items():
#         if arabic_label in label:
#             detected_labels.append(english_label)
#         elif english_label.lower() in label:
#             detected_labels.append(english_label)

#     print("Detected labels:", detected_labels)

#     # this is for duplicates values
#     detected_labels = list(set(detected_labels))

#     return detected_labels
import random
import re

from llmebench.datasets import WANLP22T3PropagandaDataset
from llmebench.models import AzureModel
from llmebench.tasks import MultilabelPropagandaTask

random.seed(1333)


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "Llama-3.1-8B-Instruct",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": WANLP22T3PropagandaDataset,
        "dataset_args": {"techniques_path": "classes.txt"},
        "task": MultilabelPropagandaTask,
        "model": AzureModel,
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


def translate_labels(label):
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
        "Obfuscation, Intentional vagueness, Confusion": (
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
        "من القائمة، اختر التقنية الدعائية الأنسب للتغريدة: "بدون تقنية"، "تشويه"، "مبالغة/تقليل"، "لغة محملة"، "النداء إلى الخوف/التحيز"، "التسمية/الملصقات"، "الشعارات"، "التكرار"، "الشك"،
        "التعمية/الغموض المتعمد/الارتباك"، "التلويح بالعلم"، "التعميمات البراقة (الفضيلة)"، "تحريف موقف شخص (رجل القش)"، "عرض بيانات غير ذات صلة (السمكة الحمراء)"، "النداء إلى السلطة"، 
        "ماذا عن"، "مغالطة الأبيض والأسود/الديكتاتورية"، "الكليشيه القاطع للفكر"، أو "التبسيط السببي"."
        """
    return [
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        }
    ]


def post_process(response):
    if not response or "error" in response or "output" not in response:
        print("Error or missing output in response:", response)
        return None

    label = re.sub(r"<[^>]+>", "", response["output"].strip().lower())
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

    detected_labels = [
        english_label
        for arabic_label, english_label in label_mapping.items()
        if arabic_label in label or english_label.lower() in label
    ]
    print("Detected labels:", list(set(detected_labels)))
    return list(set(detected_labels))
