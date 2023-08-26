import os
import re

from llmebench.datasets import ANERcorpDataset
from llmebench.models import GPTChatCompletionModel
from llmebench.tasks import NERTask


def config():
    return {
        "dataset": ANERcorpDataset,
        "dataset_args": {},
        "task": NERTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": [
                "B-PERS",
                "I-PERS",
                "B-LOC",
                "I-LOC",
                "B-ORG",
                "I-ORG",
                "B-MISC",
                "I-MISC",
            ],
            "max_tries": 150,
        },
        "general_args": {
            "data_path": "data/sequence_tagging_ner_pos_etc/NER/AnerCorp/ANERCorp_CamelLab_test.txt"
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "Assistant is a large language model trained by OpenAI.",
        },
        {
            "role": "user",
            "content": f'وصف المهمّة: أنت تعمل خبيرًا في التعرّف إلى الكيانات المسمّاة ومهمّتك هي توصيف نص عربي معيّن بتسميات الكيانات المسمّاة. فعليك تحديد أي كيانات مسمّاة موجودة في النص وتسميتها. وتسميات الكيانات المسمّاة التي ستستخدمها هي PER (للأشخاص)، وLOC (للمواقع)، وORG (للمؤسّسات)، وMISC (للكيانات المتنوّعة). وقد تواجه كيانات تتألّف من عدّة كلمات، لذا تأكّد من تسمية كلّ كلمة في الكيان بالبادئة المناسبة ("B" للكلمة الأولى من الكيان، و"I" لأي كلمة غير الكلمة الأولى). أمّا بالنسبة إلى الكلمات التي لا تشكل جزءًا من أي كيان مسمّى، فعليك الرد بـ"O".\nملاحظة: تأكّد من إصدار النواتج بشكل لائحة من العديد، على أن يتألّف كلّ عديد منها من كلمة من نص الإدخال وتسمية الكيان المسمّى المقابل لها.\n الإدخال: {input_sample.split()}'
        },
    ]


def post_process(response):
    response = response["choices"][0]["message"]["content"]
    response = response.replace("\n", "").strip()
    possible_tags = [
        "B-PER",
        "I-PER",
        "B-LOC",
        "I-LOC",
        "B-ORG",
        "I-ORG",
        "O",
        "B-MISC",
        "I-MISC",
    ]
    mapping = {
        "PER-B": "B-PER",
        "PER-I": "I-PER",
        "ORG-B": "B-ORG",
        "ORG-I": "I-ORG",
        "LOC-B": "B-LOC",
        "LOC-I": "I-LOC",
        "MISC-B": "B-MISC",
        "MISC-I": "I-MISC",
    }

    matches = re.findall(r"\((.*?)\)", response)
    if matches:
        cleaned_response = []
        for match in matches:
            elements = match.split(",")
            try:
                cleaned_response.append(elements[1])
            except:
                cleaned_response.append("O")

        cleaned_response = [
            sample.replace("'", "").strip() for sample in cleaned_response
        ]
        final_cleaned_response = []
        for elem in cleaned_response:
            if elem in possible_tags:
                final_cleaned_response.append(elem)
            elif elem in mapping:
                final_cleaned_response.append(mapping[elem])
            else:
                final_cleaned_response.append("O")
    else:
        final_cleaned_response = None
    return final_cleaned_response
