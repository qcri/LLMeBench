import os
import pandas as pd

from arabic_llm_benchmark.datasets import CovidClaimDataset
from arabic_llm_benchmark.models import GPTModel
from arabic_llm_benchmark.tasks import ClaimDetectionTask

from langchain.prompts import FewShotPromptTemplate, PromptTemplate


def config():
    return {
        "dataset": CovidClaimDataset,
        "dataset_args": {},
        "task": ClaimDetectionTask,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "gpt",
            "max_tries": 10,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/claim_covid19"
            "/CT22_arabic_1B_claim_test_gold.tsv",
            "train_data_path": "data/factuality_disinformation_harmful_content/claim_covid19/CT22_arabic_1B_claim_train.tsv"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": few_shot_prompt(input_sample)
            }
        ],
    }



def few_shot_prompt(input_sample):
    # MARAM: The whole process of creating embeddings to use through langchain should be done once !
    # The package langchain requires the api key to be set using the parameter "OPENAI_API_KEY"
    os.environ["OPENAI_API_KEY"] = os.environ["AZURE_API_KEY"]

    # MARAM: This should be done once as we do with test data loading
    # Same for example selector, it should be passed to this function to avoid loading the model for every input
    train_data_path = "data/factuality_disinformation_harmful_content/claim_covid19/CT22_arabic_1B_claim_train.tsv"
    train_data = CovidClaimDataset.load_train_data(train_data_path=train_data_path)
    example_selector = CovidClaimDataset.get_fewshot_selector(train_data)

    # MARAM: This part is asset specific to create the prompt
    # MARAM: This is asset specific, you can have the keys to be anything you want
    example_prompt = PromptTemplate(
        input_variables=["input", "label"],
        template="Input: {input}\nLabel: {label}",
    )
    mmr_prompt = FewShotPromptTemplate(
        # We provide an ExampleSelector.
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Does this sentence contain a factual claim? Answer only by yes or no.",
        suffix="Input: {input}\nLabel:",
        input_variables=["input"],
    )

    # Create few shot prompt from input_sample and few shots selected
    out_prompt = mmr_prompt.format(input=input_sample)

    return out_prompt

def post_process(response):
    pred_label = response["choices"][0]["text"]
    pred_label = pred_label.replace(".", "").strip().lower()

    if pred_label == "yes" or pred_label == "the sentence contains a factual claim":
        pred_label = "1"
    if pred_label == "no":
        pred_label = "0"

    return pred_label
