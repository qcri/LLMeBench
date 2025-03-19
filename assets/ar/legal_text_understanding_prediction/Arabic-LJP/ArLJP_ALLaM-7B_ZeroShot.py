import json
import re

from llmebench.datasets import ArabicLJPDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import NLGenerationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "ALLaM-7B-Instruct-preview",
        "description": "ALLaM is a series of powerful language models designed to advance Arabic Language Technology (ALT) developed by the National Center for Artificial Intelligence (NCAI) at the Saudi Data and AI Authority (SDAIA).",
        "scores": {"BLEU": 0.119, "METEOR": 0.168},
    }


def config():
    return {
        "dataset": ArabicLJPDataset,
        "task": NLGenerationTask,
        "model": OpenAIModel,
        "general_args": {"test_split": "test"},
    }


def prompt(input_sample):
    # Define the question prompt
    user_prompt = f"""
        {input_sample['instruction']}    
         {input_sample['input']}
        """

    # Define the assistant prompt
    system_prompt = """
     أنت محامي قانوني. بناءً على الوقائع والأسباب المقدمة، حدد واكتب نص الحكم القانوني المناسب بدقة، مع الالتزام بالمبادئ القانونية المتوقعة.
    """
    return [
        {"role": "user", "content": user_prompt},
        {"role": "system", "content": system_prompt},
    ]


def post_process(response):
    content = response["choices"][0]["message"]["content"].strip()

    return content
