from llmebench.datasets import TyDiQADataset
from llmebench.models import VLLMModel
from llmebench.tasks import QATask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Fanar 7B",
        "description": "Locally hosted Fanar 7B model using the VLLM.",
        "scores": {"F1 | exact match": "0.053"},
    }


def config():
    return {
        "dataset": TyDiQADataset,
        "task": QATask,
        "model": VLLMModel,
        "model_args": {
            "max_tries": 50,
        },
        "general_args": {"test_split": "dev"},
    }


def prompt(input_sample):
    return [
        {
            "role": "user",
            "content": (
                "مهمتك هي الإجابة على الأسئلة العربية بناء على سياق معين. يجب أن تستخرج إجاباتك من السياق."
            + "\n"
            + f"السياق: {input_sample['context']}\n"
            + f"السؤال:  {input_sample['question']}\n"
            + "الإجابة: "
            ),
        },
    ]


def post_process(response):
    if "messages" in response:
        if "content" in response["messages"]:
            label = response["messages"]["content"].strip()
            label = label.replace("<s>", "")
            label = label.replace("</s>", "")
    elif "content" in response:
        label = response["content"].strip()
        label = label.replace("<s>", "")
        label = label.replace("</s>", "")
    else:
        #print("Response .. " + str(response))
        label = ""

    label = label.replace("الجواب:", "")
    label = label.replace("الإجابة:", "")
    label = label.strip()

    return label