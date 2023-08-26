import os

from llmebench.datasets import ArSASSentimentDataset
from llmebench.models import GPTChatCompletionModel
from llmebench.tasks import SentimentTask


def config():
    return {
        "dataset": ArSASSentimentDataset,
        "dataset_args": {},
        "task": SentimentTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["Positive", "Negative", "Neutral", "Mixed"],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/sentiment/ArSAS-test.txt"
        },
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
