import json
import re

from llmebench.datasets import XLSumDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import NLGenerationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Fanar",
        "description": "FANAR-C model",
        "scores": {},
    }


def config():
    return {
        "dataset": XLSumDataset,
        "task": NLGenerationTask,
        "model": OpenAIModel,
        "general_args": {"test_split": "test"},
    }


def prompt(input_sample):
    # Define the question prompt
    user_prompt = f"""
        
        ADD INSTRUCTION    
         {input_sample['input']}

        """

    # Define the assistant prompt
    system_prompt = """
    You are an Arabic AI assistant specialized in providing detailed and accurate answers across various fields. Your task is to deliver clear, concise, and relevant information. 
    """
    return [
        {"role": "user", "content": user_prompt},
        {"role": "system", "content": system_prompt},
    ]


def post_process(response):
    content = response["choices"][0]["message"]["content"].strip()

    return content
