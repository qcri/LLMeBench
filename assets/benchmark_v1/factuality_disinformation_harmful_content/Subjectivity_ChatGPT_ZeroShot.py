from arabic_llm_benchmark.datasets import CheckWorthinessDataset
from arabic_llm_benchmark.tasks import CheckWorthinessTask
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel

import os

def config():
	return {
		"dataset": SubjectivityDataset,
		"dataset_args": {},
		"task": SubjectivityTask,
		"task_args": {"test": "useless"},
		"model": RandomGPTModel,
		"model_args": {
			"api_type": "azure",
			"api_version": "2023-03-15-preview",
			"api_base": os.environ["AZURE_API_URL"],
			"api_key": os.environ["AZURE_API_KEY"],
			"engine_name": "gpt",
			"class_labels": ["SUBJ", "OBJ"],
			"ignore_cache": True
		},
		"general_args": {
			"data_path": "/Users/firojalamqcri/QCRI/ALT_tanbih/LLM_benchmarking_arabic_nlp_tasks/exp_LLM_benchmarking_arabic_nlp_tasks/tasks/factuality_disinformation_harmful_content/subjectivity/data/dev_ar.tsv"
		}
	}

def prompt(input_sample):
	system_prompt = "## INSTRUCTION\nYou are an expert social media content analyst.\n\n"
	input_sample = f"Classify the \"Sentence\" as SUBJ or OBJ. Provide only label.\n##Sentence: {input_sample} \n## Response:\n"

	return {
		"messages": [
			{
				{"role": "system", "content": f"{system_prompt}"},
				{"role": "user", "content": f"{input_sample}"},

			}
		]
	}

def post_process(response):
	label = response["response"]["choices"][0]["message"]["content"]
	return label
