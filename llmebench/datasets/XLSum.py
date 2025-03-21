import json

import pandas as pd

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class XLSumDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(XLSumDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """
                @inproceedings{hasan-etal-2021-xl,
                    title = "{XL}-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages",
                    author = "Hasan, Tahmid  and
                    Bhattacharjee, Abhik  and
                    Islam, Md. Saiful  and
                    Mubasshir, Kazi  and
                    Li, Yuan-Fang  and
                    Kang, Yong-Bin  and
                    Rahman, M. Sohel  and
                    Shahriyar, Rifat",
                    editor = "Zong, Chengqing  and
                    Xia, Fei  and
                    Li, Wenjie  and
                    Navigli, Roberto",
                    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
                    month = aug,
                    year = "2021",
                    address = "Online",
                    publisher = "Association for Computational Linguistics",
                    url = "https://aclanthology.org/2021.findings-acl.413/",
                    doi = "10.18653/v1/2021.findings-acl.413",
                    pages = "4693--4703"
                }            
                @article{kmainasi2024llamalens,
                title="{LlamaLens: Specialized Multilingual LLM for Analyzing News and Social Media Content},
                author={Kmainasi, Mohamed Bayan and Shahroor, Ali Ezzat and Hasanain, Maram and Laskar, Sahinur Rahman and Hassan, Naeemul and Alam, Firoj},
                journal={arXiv preprint arXiv:2410.15308},
                year={2024}
                }                
            }""",
            "link": "https://huggingface.co/datasets/QCRI/LlamaLens-Arabic-Native, https://huggingface.co/datasets/csebuetnlp/xlsum",
            "license": "CC BY-NC 4.0",
            "splits": {
                "train": "train.json",
                "dev": "dev.json",
                "test": "test.json",
            },
            "task_type": TaskType.NLGenerationTask,
        }

    @staticmethod
    def get_data_sample():
        return {
            "id": "a unique id",
            "input": {
                "input": "A long text.",
                "instruction": "instruction in English",
                "native_instruction": "instruction in Arabic",
            },
            "label": "A summary",
        }

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)

        # Load JSON data into a pandas DataFrame
        df = pd.read_json(data_path, encoding="utf-8")

        # Format the data as required
        df["formatted_data"] = df.apply(
            lambda row: {
                "input": {
                    "input": row["input"],
                    "instruction": row["instruction"],
                },
                "label": row["output"],
                "line_number": row["id"],
            },
            axis=1,
        )
        formatted_data = df["formatted_data"].tolist()

        print(f"Loaded {len(formatted_data)} data samples from file!")
        return formatted_data
