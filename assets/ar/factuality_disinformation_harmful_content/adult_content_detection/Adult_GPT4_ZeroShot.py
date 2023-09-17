from llmebench.datasets import AdultDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import AdultTask


def config():
    return {
        "dataset": AdultDataset,
        "dataset_args": {},
        "task": AdultTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["ADULT", "NOT_ADULT"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    prompt_string = (
        # f'Classify whether the following Arabic sentence is appropriate for children. If not appropriate, just write "ADULT" without explaination, and if appropriate, just write "NOT_ADULT" without explaination \n {input_sample}'
        f'Given the following tweet, label it as "ADULT" or "NOT_ADULT" based on the content of the tweet.\n\n'
        f"tweet: {input_sample}\n"
        f"label: \n"
    )
    return [
        {
            "role": "system",
            "content": "You are an expert annotator, you can identify and label adult content within a tweet.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    out = response["choices"][0]["message"]["content"].replace("label: ", "")
    j = out.find(".")
    if j > 0:
        out = out[0:j]
    return out
