import os

from llmebench.datasets import AraBenchDataset
from llmebench.models import GPTModel
from llmebench.tasks import MachineTranslationTask


def config():
    sets = [
        "bible.test.mgr.0.ma",
        "bible.test.mgr.0.tn",
        "bible.test.msa.0.ms",
        "bible.test.msa.1.ms",
        "ldc_web_eg.test.lev.0.jo",
        "ldc_web_eg.test.lev.0.ps",
        "ldc_web_eg.test.lev.0.sy",
        "ldc_web_eg.test.mgr.0.tn",
        "ldc_web_eg.test.msa.0.ms",
        "ldc_web_eg.test.nil.0.eg",
        "ldc_web_lv.test.lev.0.lv",
        "madar.test.glf.0.iq",
        "madar.test.glf.0.om",
        "madar.test.glf.0.qa",
        "madar.test.glf.0.sa",
        "madar.test.glf.0.ye",
        "madar.test.glf.1.iq",
        "madar.test.glf.1.sa",
        "madar.test.glf.2.iq",
        "madar.test.lev.0.jo",
        "madar.test.lev.0.lb",
        "madar.test.lev.0.pa",
        "madar.test.lev.0.sy",
        "madar.test.lev.1.jo",
        "madar.test.lev.1.sy",
        "madar.test.mgr.0.dz",
        "madar.test.mgr.0.ly",
        "madar.test.mgr.0.ma",
        "madar.test.mgr.0.tn",
        "madar.test.mgr.1.ly",
        "madar.test.mgr.1.ma",
        "madar.test.mgr.1.tn",
        "madar.test.msa.0.ms",
        "madar.test.nil.0.eg",
        "madar.test.nil.0.sd",
        "madar.test.nil.1.eg",
        "madar.test.nil.2.eg",
    ]

    configs = []
    for testset in sets:
        configs.append(
            {
                "name": testset,
                "config": {
                    "dataset": AraBenchDataset,
                    "dataset_args": {
                        "src": f"{testset}.ar",
                        "tgt": f"{testset}.en",
                    },
                    "task": MachineTranslationTask,
                    "task_args": {},
                    "model": GPTModel,
                    "model_args": {
                        "api_type": "azure",
                        "api_version": "2023-03-15-preview",
                        "api_base": os.environ["AZURE_API_URL"],
                        "api_key": os.environ["AZURE_API_KEY"],
                        "engine_name": os.environ["ENGINE_NAME"],
                        "max_tries": 5,
                    },
                    "general_args": {"data_path": "data/MT/"},
                },
            }
        )

    return configs


def prompt(input_sample):
    return {
        "system_message": "You are an expert translator specialized in translating texts from Arabic to English. You are concise as you only output the translation of the text without any illustrations or extra details",
        "messages": [
            {
                "sender": "user",
                "text": f"Translate the following text to English.\nText: {input_sample}\nTranslation: ",
            }
        ],
    }


def post_process(response):
    response = response["choices"][0]["text"]
    response = response.replace('"', "")
    response = response.strip()
    return response