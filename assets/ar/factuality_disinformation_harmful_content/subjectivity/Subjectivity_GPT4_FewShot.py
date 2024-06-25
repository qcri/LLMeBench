from llmebench.datasets import ThatiARDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SubjectivityTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-8k",
        "description": "GPT4 8k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Macro-F1": "0.745"},
    }


def config():
    return {
        "dataset": ThatiARDataset,
        "task": SubjectivityTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["SUBJ", "OBJ"],
            "max_tries": 30,
        },
        "general_args": {
            "test_split": "ar/test",
            "fewshot": {"train_split": "ar/train"},
        },
    }


def prompt(input_sample, examples):
    base_prompt = f"صنف الجملة التالية لأحد النوعين: جملة ذاتية أو جملة موضوعية< \n الجُمل الذاتية تعبر عن مشاعر أو تذوق أدبي أو تفسير شخصي للمواضيع والأحداث، أما الجمل الموضوعية فتعرض حقائق وأحداث ومواضيع مبنية على بيانات واقعية. \n\n"
    return [
        {
            "role": "system",
            "content": "أنت خبير في تصنيف النصوص، ويمكنك تحليل المعلومات الموجودة في الجملة وتحديد ما إذا كانت الجملة ذاتية أو موضوعية.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    out_prompt = out_prompt + "هذه بعض الأمثلة:\n\n"
    for index, example in enumerate(examples):
        label = "جملة موضوعية" if example["label"] == "OBJ" else "جملةذاتية"

        out_prompt = (
            out_prompt
            + "مثال "
            + str(index)
            + ":"
            + "\n"
            + "الجملة: "
            + example["input"]
            + "\nالتصنيف: "
            + label
            + "\n\n"
        )

    out_prompt = out_prompt + "الجملة: " + input_sample + "\nالتصنيف: \n"

    return out_prompt


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()

    if "التصنيف: جملةموضوعية" in label:
        label_fixed = "OBJ"
    elif "التصنيف: جملة ذاتية" in label:
        label_fixed = "SUBJ"
    elif label == "موضوعية" or label == "موضوعية.":
        label_fixed = "OBJ"

    elif label == "ذاتية" or label == "ذاتية.":
        label_fixed = "SUBJ"

    return label_fixed
