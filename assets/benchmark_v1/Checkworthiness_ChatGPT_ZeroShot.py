from arabic_llm_benchmark.datasets import CheckWorthinessDataset
from arabic_llm_benchmark.tasks import CheckWorthinessTask
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel

import os

def config():
	return {
		"dataset": CheckWorthinessDataset,
		"dataset_args": {},
		"task": CheckWorthinessTask,
		"task_args": {"test": "useless"},
		"model": RandomGPTModel,
		"model_args": {
			"api_type": "azure",
			"api_version": "2023-03-15-preview",
			"api_base": os.environ["AZURE_API_URL"],
			"api_key": os.environ["AZURE_API_KEY"],
			"engine_name": "gpt",
			"class_labels": ["checkworthy", "not_checkworthy"],
			"ignore_cache": True
		},
		"general_args": {
			"data_path": "/Users/firojalamqcri/QCRI/ALT_tanbih/LLM_benchmarking_arabic_nlp_tasks/exp_LLM_benchmarking_arabic_nlp_tasks/tasks/factuality_disinformation_harmful_content/checkworthyness/data/CT22_arabic_1A_checkworthy_test_gold.tsv"
		}
	}

def prompt(input_sample):
	return {
		"system_message": "You are an AI assistant that helps people find information.",
		"messages": [
			{
				"sender":"user",
				"text": f"Classify the \"tweet\" as checkworthy or not_checkworthy. Provide only label.\n\nsentence: {input_sample} \nlabel:"
			}
		]
	}

def post_process(response):
	label = response["choices"][0]['text']

	if (label == "checkworthy"):
		label_fixed = "1"
	elif (label == "Not_checkworthy." or label == "not_checkworthy"):
		label_fixed = "0"

	return label_fixed
