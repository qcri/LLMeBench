import json

from pathlib import Path

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class FinePropSpanDataset(DatasetBase):
    def __init__(self, techniques_path=None, **kwargs):
        # Get the path to the file listing the target techniques
        self.techniques_path = Path(techniques_path) if techniques_path else None
        super(FinePropSpanDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "en",
            "citation": """
                @inproceedings{da-san-martino-etal-2019-fine,
                    title = "Fine-Grained Analysis of Propaganda in News Articles",
                    author = "Da San Martino, Giovanni  and
                    Yu, Seunghak  and
                    Barrn-Cedeno, Alberto  and
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
            "license": "Research Purpose Only",
            "splits": {
                "train": "FinePropSpan_train.jsonl",
                "dev": "FinePropSpan_dev.jsonl",
                "test": "FinePropSpan_test.jsonl",
            },
            "task_type": TaskType.MultiLabelClassification,
            "class_labels": [
                "no technique",
                "Smears",
                "Exaggeration/Minimisation",
                "Loaded Language",
                "Appeal to fear/prejudice",
                "Name calling/Labeling",
                "Slogans",
                "Repetition",
                "Doubt",
                "Obfuscation, Intentional vagueness, Confusion",
                "Flag-waving",
                "Glittering generalities (Virtue)",
                "Misrepresentation of Someone's Position (Straw Man)",
                "Presenting Irrelevant Data (Red Herring)",
                "Appeal to authority",
                "Whataboutism",
                "Black-and-white Fallacy/Dictatorship",
                "Thought-terminating cliché",
                "Causal Oversimplification",
            ],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "Tweet", "label": ["no technique"]}

    def get_predefined_techniques(self):
        # Load a pre-defined list of propaganda techniques, if available
        if self.techniques_path and self.techniques_path.exists():
            self.techniques_path = self.resolve_path(self.techniques_path)
            with open(self.techniques_path, "r", encoding="utf-8") as f:
                techniques = [label.strip() for label in f.readlines()]
        else:
            techniques = [
                "no technique",
                "Smears",
                "Exaggeration/Minimisation",
                "Loaded Language",
                "Appeal to fear/prejudice",
                "Name calling/Labeling",
                "Slogans",
                "Repetition",
                "Doubt",
                "Obfuscation, Intentional vagueness, Confusion",
                "Flag-waving",
                "Glittering generalities (Virtue)",
                "Misrepresentation of Someone's Position (Straw Man)",
                "Presenting Irrelevant Data (Red Herring)",
                "Appeal to authority",
                "Whataboutism",
                "Black-and-white Fallacy/Dictatorship",
                "Thought-terminating cliché",
                "Causal Oversimplification",
            ]

        return techniques

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)

        data = []
        with open(data_path, mode="r", encoding="utf-8") as infile:
            json_data = json.load(infile)
            for index, tweet in enumerate(json_data):
                text = tweet["text"]
                label = tweet["labels"]
                data.append({"input": text, "label": label, "line_number": index})

        return data
