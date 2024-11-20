from llmebench.datasets import ThatiARDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SubjectivityTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. Uses an prompt specified in Arabic.",
        "scores": {"Macro-F1": "0.725"},
    }


def config():
    return {
        "dataset": ThatiARDataset,
        "task": SubjectivityTask,
        "model": OpenAIModel,
    }


def prompt(input_sample):
    prompt_string = (
        f"صنّف الجملة التالية لأحد النوعين: جملة ذاتية أو جملة موضوعية.\nالجُمَل الذاتية تعبر عن مشاعر أو تذوّق أدبي أو تفسير شخصي للمواضيع والأحداث، أما الجُمَل الموضوعية فتعرض حقائق وأحداث ومواضيع مبنية على بيانات واقعية.\n\n"
        f"الجملة: {input_sample}\n"
        f"التصنيف: \n"
    )
    return [
        {
            "role": "system",
            "content": "أنت خبير في تصنيف النصوص، ويمكنك تحليل المعلومات الموجودة في الجملة وتحديد ما إذا كانت الجملة ذاتية أو موضوعية",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()

    if "ﺎﻠﺘﺼﻨﻴﻓ: ﺞﻤﻟﺔﻣﻮﺿﻮﻌﻳﺓ" in label:
        label_fixed = "OBJ"
    elif "ﺎﻠﺘﺼﻨﻴﻓ: ﺞﻤﻟﺓ ﺫﺎﺘﻳﺓ" in label:
        label_fixed = "SUBJ"
    elif label == "ﻡﻮﺿﻮﻌﻳﺓ" or label == "ﻡﻮﺿﻮﻌﻳﺓ.":
        label_fixed = "OBJ"

    elif label == "ﺫﺎﺘﻳﺓ" or label == "ﺫﺎﺘﻳﺓ.":
        label_fixed = "SUBJ"

    return label_fixed


def post_process_old(response):
    label = response["choices"][0]["message"]["content"].lower()

    if "لاموضوعية" in label:
        label_fixed = "SUBJ"
    elif "ذاتية" in label:
        label_fixed = "SUBJ"
    elif (
        label == "موضوعية" or label == "التصنيف: موضوعية" or "التصنيف: موضوعية" in label
    ):
        label_fixed = "OBJ"
    else:
        label_fixed = None

    return label_fixed
