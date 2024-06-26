import json

from llmebench.datasets import MultiNativQADataset
from llmebench.models import AzureModel
from llmebench.tasks import MultiNativQATask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Mistral 7b",
        "description": "Deployed on Azure.",
        "scores": {},
    }


def config():
    return {
        "dataset": MultiNativQADataset,
        "task": MultiNativQATask,
        "model": AzureModel,
    }


def prompt(input_sample):

    # Define the question prompt
    # Provide your response in the following JSON format and provide JSON output only. No additional text.
    question_prompt = f"""
    Please use your expertise to answer the following Arabic Question. You should 'Answer' only in Arabic. 
    

    Question: {input_sample}

    Answer: 
    """

    # Define the assistant prompt
    assistant_prompt = """
    I am an Arabic AI assistant specialized in providing detailed and accurate answers across various fields.
    """

    return [
        {
            "role": "user",
            "content": question_prompt,
        },
        {
            "role": "assistant",
            "content": assistant_prompt,
        },
    ]


def post_process(response):
    data = response["output"]
    response = json.loads(data)
    answer = response["answer"]
    return answer
