import os 

from arabic_llm_benchmark.datasets import MLQADataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import QATask


def config():
    return {
        "dataset": MLQADataset,
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
            "max_tries": 50,
        },
        "general_args": {"data_path":  "data/QA/mlqa/test-context-ar-question-ar.json"},
    }




def prompt(input_sample): 

    return [
        {
            "role": "system",
            "content": "Assistant is a large language model trained by OpenAI.",
        },
        {
            "role": "user",
            "content":f"Your task is to answer questions in Arabic based on a given context.\nNote: Your answers should be spans extracted from the given context without any illustrations.\nYou don't need to provide a complete answer\nContext:{input_sample['context']}\nQuestion:{input_sample['question']}\nAnswer:",
        },
    ]



def post_process(response): 
    return response["choices"][0]["message"]["content"]

