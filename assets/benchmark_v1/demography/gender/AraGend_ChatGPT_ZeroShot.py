from arabic_llm_benchmark.datasets import ArabGendDataset
from arabic_llm_benchmark.tasks import DemographyGenderTask
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel

import os

def config():
	return {
		"dataset": ArabGendDataset,
		"dataset_args": {},
		"task": DemographyGenderTask,
		"task_args": {},
		"model": RandomGPTModel,
		"model_args": {
			"api_type": "azure",
			"api_version": "2023-03-15-preview",
			"api_base": os.environ["AZURE_API_URL"],
			"api_key": os.environ["AZURE_API_KEY"],
			"engine_name": "gpt",
			"class_labels": ["m", "f"],
		},
		"general_args": {
			"data_path": "/Users/fahim/QCRI/Projects/2023-LLM-benchmarking-for-arabic/packaging/raw/demography/gender/data/gender-test.txt",
			"ignore_cache": True
		}
	}

def prompt(input_sample):
	return {
		"system_message": "You are an AI assistant that helps people find information.",
		"messages": [
			{
				"sender":"user",
				"text": f"If the following person name can be considered as male, write 'm' without explnanation, and if it can be considered as female, write 'f' without explnanation.\n {input_sample}"
			}
		]
	}

def post_process(response):
	return response["choices"][0]['text']
