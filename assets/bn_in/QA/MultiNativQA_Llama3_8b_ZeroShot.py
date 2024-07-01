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
        "general_args": {"test_split": "bangla_in"},
    }


def prompt(input_sample):

    # Define the question prompt
    question_prompt = f"""
    Please use your expertise to answer the following Bangla question. You should only Answer in Bengali, West Bengal. 
    Provide your response in the following JSON format: {{"answer": "your answer"}}. 
    Please provide JSON output only. No additional text.

    Question: {input_sample}
    Answer: 
    """

    # Define the assistant prompt
    assistant_prompt = """
    I am a Bengali, West Bengal AI assistant specialized in providing detailed and accurate answers across various fields. I aim to deliver clear, concise, and relevant information. How can I assist you today?
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
