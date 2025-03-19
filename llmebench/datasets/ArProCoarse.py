import json
from pathlib import Path

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class ArProCoarseDataset(DatasetBase):
    def __init__(self, techniques_path=None, **kwargs):
        # Get the path to the file listing the target techniques
        self.techniques_path = Path(techniques_path) if techniques_path else None
        super(ArProCoarseDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """
                @inproceedings{hasanain-etal-2024-gpt,
                    title = "Can GPT-4 Identify Propaganda? Annotation and Detection of Propaganda Spans in News Articles",
                    author = "Hasanain, Maram  and
                    Ahmad, Fatema  and
                    Alam, Firoj",
                    editor = "Calzolari, Nicoletta  and
                    Kan, Min-Yen  and
                    Hoste, Veronique  and
                    Lenci, Alessandro  and
                    Sakti, Sakriani  and
                    Xue, Nianwen",
                    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
                    month = may,
                    year = "2024",
                    address = "Torino, Italia",
                    publisher = "ELRA and ICCL",
                    url = "https://aclanthology.org/2024.lrec-main.244/",
                    pages = "2724--2744",
                    abstract = "The use of propaganda has spiked on mainstream and social media, aiming to manipulate or mislead users. While efforts to automatically detect propaganda techniques in textual, visual, or multimodal content have increased, most of them primarily focus on English content. The majority of the recent initiatives targeting medium to low-resource languages produced relatively small annotated datasets, with a skewed distribution, posing challenges for the development of sophisticated propaganda detection models. To address this challenge, we carefully develop the largest propaganda dataset to date, ArPro, comprised of 8K paragraphs from newspaper articles, labeled at the text span level following a taxonomy of 23 propagandistic techniques. Furthermore, our work offers the first attempt to understand the performance of large language models (LLMs), using GPT-4, for fine-grained propaganda detection from text. Results showed that GPT-4`s performance degrades as the task moves from simply classifying a paragraph as propagandistic or not, to the fine-grained task of detecting propaganda techniques and their manifestation in text. Compared to models fine-tuned on the dataset for propaganda detection at different classification granularities, GPT-4 is still far behind. Finally, we evaluate GPT-4 on a dataset consisting of six other languages for span detection, and results suggest that the model struggles with the task across languages. We made the dataset publicly available for the community."
                }               
            """,
            "link": "https://huggingface.co/datasets/QCRI/ArmPro/",
            "license": " https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "splits": {
                "test": "ArMPro_coarse_test.jsonl",
                "dev": "ArMPro_coarse_dev.jsonl",
                "train": "ArMPro_coarse_train.jsonl",
            },
            "task_type": TaskType.MultiLabelClassification,
            "class_labels": [
                "Manipulative_Wording",
                "no_technique",
                "Reputation",
                "Justification",
                "Simplification",
                "Call",
                "Distraction",
            ],
        }

    @staticmethod
    def get_data_sample():
        return {
            "id": "001",
            "input": "paragraph",
            "label": ["no_technique"],
            "type": "paragraph",
        }

    def get_predefined_techniques(self):
        # Load a pre-defined list of propaganda techniques, if available
        if self.techniques_path and self.techniques_path.exists():
            self.techniques_path = self.resolve_path(self.techniques_path)
            with open(self.techniques_path, "r", encoding="utf-8") as f:
                techniques = [label.strip() for label in f.readlines()]
        else:
            techniques = [
                "Manipulative_Wording",
                "no_technique",
                "Reputation",
                "Justification",
                "Simplification",
                "Call",
                "Distraction",
            ]

        return techniques

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)

        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                line_data = json.loads(line)
                id = line_data.get("paragraph_id", None)
                text = line_data.get("paragraph", "")
                label = line_data.get("labels", "")
                if len(label) == 0:
                    label = ["no_technique"]
                data.append({"input": text, "label": label, "line_number": id})

        return data
