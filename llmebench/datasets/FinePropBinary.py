import json

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class FinePropBinaryDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(FinePropBinaryDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "en",
            "citation": """
                @inproceedings{da-san-martino-etal-2019-fine,
                    title = "Fine-Grained Analysis of Propaganda in News Articles",
                    author = "Da San Martino, Giovanni  and
                    Yu, Seunghak  and
                    Barron-Cedeno, Alberto  and
                    Petrov, Rostislav  and
                    Nakov, Preslav",
                    editor = "Inui, Kentaro  and
                    Jiang, Jing  and
                    Ng, Vincent  and
                    Wan, Xiaojun",
                    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
                    month = nov,
                    year = "2019",
                    address = "Hong Kong, China",
                    publisher = "Association for Computational Linguistics",
                    url = "https://aclanthology.org/D19-1565/",
                    doi = "10.18653/v1/D19-1565",
                    pages = "5636--5646",
                    abstract = "Propaganda aims at influencing people`s mindset with the purpose of advancing a specific agenda. Previous work has addressed propaganda detection at document level, typically labelling all articles from a propagandistic news outlet as propaganda. Such noisy gold labels inevitably affect the quality of any learning system trained on them. A further issue with most existing systems is the lack of explainability. To overcome these limitations, we propose a novel task: performing fine-grained analysis of texts by detecting all fragments that contain propaganda techniques as well as their type. In particular, we create a corpus of news articles manually annotated at fragment level with eighteen propaganda techniques and propose a suitable evaluation measure. We further design a novel multi-granularity neural network, and we show that it outperforms several strong BERT-based baselines."
                }
                @inproceedings{piskorski-etal-2023-semeval,
                    title = "SemEval-2023 Task 3: Detecting the Category, the Framing, and the Persuasion Techniques in Online News in a Multi-lingual Setup",
                    author = "Piskorski, Jakub  and
                    Stefanovitch, Nicolas  and
                    Da San Martino, Giovanni  and
                    Nakov, Preslav",
                    editor = {Ojha, Atul Kr.  and
                    Dogruoz, A. Seza  and
                    Da San Martino, Giovanni  and
                    Tayyar Madabushi, Harish  and
                    Kumar, Ritesh  and
                    Sartori, Elisa},
                    booktitle = "Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023)",
                    month = jul,
                    year = "2023",
                    address = "Toronto, Canada",
                    publisher = "Association for Computational Linguistics",
                    url = "https://aclanthology.org/2023.semeval-1.317/",
                    doi = "10.18653/v1/2023.semeval-1.317",
                    pages = "2343--2361",
                    abstract = "We describe SemEval-2023 task 3 on Detecting the Category, the Framing, and the Persuasion Techniques in Online News in a Multilingual Setup: the dataset, the task organization process, the evaluation setup, the results, and the participating systems. The task focused on news articles in nine languages (six known to the participants upfront: English, French, German, Italian, Polish, and Russian), and three additional ones revealed to the participants at the testing phase: Spanish, Greek, and Georgian). The task featured three subtasks: (1) determining the genre of the article (opinion, reporting, or satire), (2) identifying one or more frames used in an article from a pool of 14 generic frames, and (3) identify the persuasion techniques used in each paragraph of the article, using a taxonomy of 23 persuasion techniques. This was a very popular task: a total of 181 teams registered to participate, and 41 eventually made an official submission on the test set."
                }                           
            """,
            "link": "",
            "license": "",
            "splits": {
                "test": "FinePropBinary_test.jsonl",
                "dev": "FinePropBinary_dev.jsonl",
                "train": "FinePropBinary_train.jsonl",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["true", "false"],
        }

    @staticmethod
    def get_data_sample():
        return {"id": "001", "input": "paragraph", "label": "true", "type": "paragraph"}

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)

        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                line_data = json.loads(line)
                id = line_data.get("paragraph_id") or line_data.get("tweet_id")
                text = line_data.get("paragraph") or line_data.get("text")
                label = line_data.get("label", "").lower()
                data.append({"input": text, "label": label, "line_number": id})

        return data
