from llmebench.datasets import ARCDDataset
from llmebench.models import FastChatModel
from llmebench.tasks import QATask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "JAIS-13b",
        "description": "Locally hosted JAIS-13b-chat model using FastChat.",
        "scores": {"Macro-F1": ""},
    }


def config():
    return {
        "dataset": ARCDDataset,
        "task": QATask,
        "model": FastChatModel,
    }


def prompt(input_sample):
    base_prompt = (
        f"مهمتك هي الإجابة على الأسئلة باللغة العربية بناءً على سياق معين.\nملاحظة: يجب أن تكون إجاباتك مستخرجة من السياق المحدد دون أي اضافات.\nلست بحاجة إلى تقديم إجابة كاملة.\nالسياق: {input_sample['context']}\n السؤال: {input_sample['question']}\n الجواب:"
    )
    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    return response["choices"][0]["message"]["content"]
