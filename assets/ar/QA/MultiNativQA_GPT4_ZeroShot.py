from llmebench.datasets import MultiNativQADataset
from llmebench.models import OpenAIModel
from llmebench.tasks import MultiNativQATask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Llama3 8b",
        "description": "Deployed on Azure.",
        "scores": {},
    }


def config():
    return {
        "dataset": MultiNativQADataset,
        "task": MultiNativQATask,
        "model": OpenAIModel,
    }


def prompt(input_sample):

    # Define the question prompt
    question_prompt = f"""
    Please use your expertise to answer the following Arabic question. Answer in Arabic and rate your confidence level from 1 to 10. Provide your response in the following JSON format: {{"answer": "your answer", "score": your confidence score}}. Please provide JSON output only. No additional text.

    Question: {input_sample}
    
    """

    # Define the assistant prompt
    assistant_prompt = """
    I am an Arabic AI assistant specialized in providing detailed and accurate answers across various fields. I aim to deliver clear, concise, and relevant information. How can I assist you today?
    """
    return [
        {"role": "user", "content": question_prompt},
        {"role": "assistant", "content": assistant_prompt},
    ]


def post_process(response):
    return response["choices"][0]["message"]["content"]
