import json
import re

from llmebench.datasets import MultiNativQADataset
from llmebench.models import OpenAIModel
from llmebench.tasks import MultiNativQATask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "GPT4-o",
        "description": "Deployed on Azure.",
        "scores": {},
    }


def config():
    return {
        "dataset": MultiNativQADataset,
        "task": MultiNativQATask,
        "model": OpenAIModel,
        "general_args": {"test_split": "arabic_qa"},
    }


def prompt(input_sample):
    # Define the question prompt
    question_prompt = f"""
        Please use your expertise to answer the following Arabic question. Answer in Arabic. Please provide Answer only. No additional text. Answer should be limited to less or equal to {input_sample['length']} words.

        Question: {input_sample['question']}

        """

    # Define the assistant prompt
    assistant_prompt = """
    You are an Arabic AI assistant specialized in providing detailed and accurate answers across various fields. Your task is to deliver clear, concise, and relevant information. 
    """
    return [
        {"role": "user", "content": question_prompt},
        {"role": "assistant", "content": assistant_prompt},
    ]


def post_process(response):
    content = response["choices"][0]["message"]["content"].strip()
    return content
