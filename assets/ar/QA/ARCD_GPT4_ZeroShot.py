import os

from llmebench.datasets import ARCDDataset
from llmebench.models import GPTChatCompletionModel
from llmebench.tasks import QATask


def config():
    return {
        "dataset": ARCDDataset,
        "dataset_args": {},
        "task": QATask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": "NA",
            "max_tries": 150,
        },
        "general_args": {"data_path": "data/QA/arcd/arcd-test.json"},
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "Assistant is a large language model trained by OpenAI.",
        },
        {
            "role": "user",
            "content": f"مهمتك هي الإجابة على الأسئلة باللغة العربية بناءً على سياق معين.\nملاحظة: يجب أن تكون إجاباتك مستخرجة من السياق المحدد دون أي اضافات.\nلست بحاجة إلى تقديم إجابة كاملة.\nالسياق: {input_sample['context']}\n السؤال: {input_sample['question']}\n الجواب:"
        },
    ]


def post_process(response):
    return response["choices"][0]["message"]["content"]