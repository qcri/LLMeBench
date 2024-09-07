import json

from llmebench.datasets import MultiNativQADataset
from llmebench.models import AzureModel
from llmebench.tasks import MultiNativQATask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "LLama 3 8b",
        "description": "Deployed on Azure.",
        "scores": {},
    }


def config():
    return {
        "dataset": MultiNativQADataset,
        "task": MultiNativQATask,
        "model": AzureModel,
        "general_args": {"test_split": "arabic_qa"},
    }


def prompt(input_sample):

    # Define the question prompt
    question_prompt = f"""
    Please use your expertise to answer the following Arabic question. Answer in Arabic and rate your confidence level from 1 to 10.
    Provide your response in the following JSON format: {{"answer": "your answer", "score": your confidence score}}.
    Please provide JSON output only. No additional text. Answer should be limited to less or equal to {input_sample['length']} words.

    Question: {input_sample['question']}
    """

    # Define the assistant prompt
    assistant_prompt = """
    You are an Arabic AI assistant specialized in providing detailed and accurate answers across various fields.
    Your task is to deliver clear, concise, and relevant information. 
    """

    return [
        {
            "role": "assistant",
            "content": assistant_prompt,
        },
        {
            "role": "user",
            "content": question_prompt,
        },
    ]


def post_process(response):
    data = response["output"]
    response = json.loads(data)
    answer = response["answer"]
    return answer
