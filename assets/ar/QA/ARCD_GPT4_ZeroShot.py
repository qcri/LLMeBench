from llmebench.datasets import ARCDDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import QATask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. Uses an prompt specified in Arabic.",
        "scores": {"F1": "0.654"},
    }


def config():
    return {
        "dataset": ARCDDataset,
        "task": QATask,
        "model": OpenAIModel,
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "Assistant is a large language model trained by OpenAI.",
        },
        {
            "role": "user",
            "content": f"مهمتك هي الإجابة على الأسئلة باللغة العربية بناءً على سياق معين.\nملاحظة: يجب أن تكون إجاباتك مستخرجة من السياق المحدد دون أي اضافات.\nلست بحاجة إلى تقديم إجابة كاملة.\nالسياق: {input_sample['context']}\n السؤال: {input_sample['question']}\n الجواب:",
        },
    ]


def post_process(response):
    return response["choices"][0]["message"]["content"]
