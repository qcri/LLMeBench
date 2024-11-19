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
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
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

def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n\n"
    out_prompt = out_prompt + "Here are some examples:\n\n"
    for index, example in enumerate(examples):
        tech_str = ""
        for t in example["label"]:
            tech_str += "'" + t + "', "

        out_prompt = (
            out_prompt
            + "Example "
            + str(index)
            + ":"
            + "\n"
            + "Tweet: "
            + example["input"]
            + "\nLabel: "
            + tech_str[:-2]  # Remove the trailing comma and space
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "Tweet: " + input_sample + "\nLabel: \n"

    return out_prompt


def prompt(input_sample, examples):
    instruction = """
Review the following tweets and analyze the propaganda techniques used. Choose only the applicable categories from: "no technique," "Smears,"
 "Exaggeration/Minimisation," "Loaded Language," "Appeal to fear/prejudice," "Name calling/Labeling," "Slogans," "Repetition," "Doubt,"
 "Obfuscation/Intentional vagueness/Confusion," "Flag-waving," "Glittering generalities (Virtue),"
 "Misrepresentation of Someone's Position (Straw Man)," "Presenting Irrelevant Data (Red Herring)," "Appeal to authority,"
"Whataboutism," "Black-and-white Fallacy/Dictatorship," "Thought-terminating cliché," or "Causal Oversimplification."
    """
    base_prompt = instruction.strip()
    
    return [
        {
            "role": "user",
            "content": (
                few_shot_prompt(input_sample, base_prompt, examples)
            ),
        }
    ]

def post_process(response):
    if not response or 'error' in response or 'output' not in response:
        print("Error or missing output in response:", response)
        return "No respose"  # Safely default to NOT_ADULT when unsure

    label = response["output"].strip().lower()
    label = label.replace("<s>", "").replace("</s>", "")
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
        "التعمية/الغموض المتعمد/الارتباك": "Obfuscation, Intentional vagueness, Confusion",
        "التلويح بالعلم": "Flag-waving",
        "التعميمات البراقة (الفضيلة)": "Glittering generalities (Virtue)",
        "تحريف موقف شخص (رجل القش)": "Misrepresentation of Someone's Position (Straw Man)",
        "عرض بيانات غير ذات صلة (السمكة الحمراء)": "Presenting Irrelevant Data (Red Herring)",
        "النداء إلى السلطة": "Appeal to authority",
        "ماذا عن": "Whataboutism",
        "مغالطة الأبيض والأسود/الديكتاتورية": "Black-and-white Fallacy/Dictatorship",
        "الكليشيه القاطع للفكر": "Thought-terminating cliché",
        "التبسيط السببي": "Causal Oversimplification"
    }

    detected_labels = []
    for arabic_label, english_label in label_mapping.items():
        if arabic_label in label:
            detected_labels.append(english_label)
        elif english_label.lower() in label:
            detected_labels.append(english_label)

    print("Detected labels:", detected_labels) 

    # Remove duplicates
    detected_labels = list(set(detected_labels))

    return detected_labels
