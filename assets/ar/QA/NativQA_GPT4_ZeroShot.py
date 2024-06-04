from llmebench.datasets import NativQADataset
from llmebench.models import OpenAIModel
from llmebench.tasks import NativQATask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-o (version 2024-02-15-preview)",
        "description": "GPT4 o model hosted on Azure, using the ChatCompletion API. API version '2024-02-15-preview'. Uses an prompt specified in English.",
        "scores": {},
    }


def config():
    return {
        "dataset": NativQADataset,
        "task": NativQATask,
        "model": OpenAIModel,
    }


def prompt(input_sample):

    prompt = f"Using your expertise, please answer the question below. How confident are you in your answer? Score it on a scale of 1 to 10. Provide the output in a JSON format: {{'answer': 'text', 'score': 10}}\n\nquestion: {input_sample}"

    return [
        {
            "role": "system",
            "content": "You are an expert with knowledge about regional and local information all over the world.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def post_process(response):
    return response["choices"][0]["message"]["content"]
