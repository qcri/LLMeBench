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

	## GPT 3.5 - turbo
	# return {
	# 	"system_message": "You are an AI assistant that helps people find information.",
	# 	"messages": [
	# 		{
	# 			"sender":"user",
	# 			"text": f"Classify the \"tweet\" as checkworthy or not_checkworthy. Provide only label.\n\nsentence: {input_sample} \nlabel:"
	# 		}
	# 	]
	# }

	system_prompt = "## INSTRUCTION\nYou are an expert social media content analyst.\n\n"
	input_sample = f"Classify the \"Tweet\" as SUBJ or OBJ. Provide only label.\n## Tweet: {input_sample} \n## Response:\n"

	return {
		"messages": [
			{
				{"role": "system", "content": f"{system_prompt}"},
				{"role": "user", "content": f"{input_sample}"},

			}
		]
	}


def post_process(response):
	# label = response["choices"][0]['text']
	label = response["response"]["choices"][0]["message"]["content"]

	if (label == "checkworthy"):
		label_fixed = "1"
	elif (label == "Not_checkworthy." or label == "not_checkworthy"):
		label_fixed = "0"

	return label_fixed
