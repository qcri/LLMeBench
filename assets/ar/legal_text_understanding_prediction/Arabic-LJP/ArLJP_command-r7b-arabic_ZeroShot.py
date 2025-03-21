import json
import re

from llmebench.datasets import ArabicLJPDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import NLGenerationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "c4ai-command-r7b-arabic-02-2025",
        "description": "C4AI Command R7B Arabic is an open weights research release of a 7 billion parameter custom model with advanced capabilities optimized for the Arabic language (MSA dialect) along with English.",
        "scores": {"BLEU": 0.0909, "METEOR": 0.149},
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
