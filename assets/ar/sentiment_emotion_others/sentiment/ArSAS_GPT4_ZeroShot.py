from llmebench.datasets import ArSASDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SentimentTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. Uses an prompt specified in Arabic.",
        "scores": {"Macro-F1": "0.547"},
    }


def config():
    return {
        "dataset": ArSASDataset,
        "task": SentimentTask,
        "model": OpenAIModel,
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": f'وصف المهمّة: مهمّتك هي تحديد الشعور الذي يعكسه نص معيّن باللغة العربيّة. وسيكون الإدخال نصًا باللغة العربيّة بينما يمكن أن تتمثّل المخرجات بواحدة من الفئات التالية ."Positive", "Negative", "Neutral", "Mixed": ملاحظة: يجب أن تنحسر المخرجات بواحدة من الفئات المذكورة من دون أي توضيح أو تفاصيل إضافيّة.\nInput: {input_sample}\nLabel: ',
        },
    ]


def post_process(response):
    out = response["choices"][0]["message"]["content"]
    j = out.find(".")
    if j > 0:
        out = out[0:j]
    return out
