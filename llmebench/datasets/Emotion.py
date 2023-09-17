from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class EmotionDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(EmotionDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{hassan-etal-2022-cross,
                title = "Cross-lingual Emotion Detection",
                author = "Hassan, Sabit  and
                  Shaar, Shaden  and
                  Darwish, Kareem",
                booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
                month = jun,
                year = "2022",
                address = "Marseille, France",
                publisher = "European Language Resources Association",
                url = "https://aclanthology.org/2022.lrec-1.751",
                pages = "6948--6958",
                abstract = "Emotion detection can provide us with a window into understanding human behavior. Due to the complex dynamics of human emotions, however, constructing annotated datasets to train automated models can be expensive. Thus, we explore the efficacy of cross-lingual approaches that would use data from a source language to build models for emotion detection in a target language. We compare three approaches, namely: i) using inherently multilingual models; ii) translating training data into the target language; and iii) using an automatically tagged parallel corpus. In our study, we consider English as the source language with Arabic and Spanish as target languages. We study the effectiveness of different classification models such as BERT and SVMs trained with different features. Our BERT-based monolingual models that are trained on target language data surpass state-of-the-art (SOTA) by 4{\\%} and 5{\\%} absolute Jaccard score for Arabic and Spanish respectively. Next, we show that using cross-lingual approaches with English data alone, we can achieve more than 90{\\%} and 80{\\%} relative effectiveness of the Arabic and Spanish BERT models respectively. Lastly, we use LIME to analyze the challenges of training cross-lingual models for different language pairs.",
            }""",
            "link": "https://competitions.codalab.org/competitions/17751",
            "license": "Restricted",
            "download_url": "http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip",
            "splits": {
                "test": "test-gold.txt",
                "train": "train.txt",
            },
            "task_type": TaskType.MultiLabelClassification,
            "class_labels": [
                "anger",
                "disgust",
                "fear",
                "joy",
                "love",
                "optimism",
                "pessimism",
                "sadness",
                "surprise",
                "trust",
            ],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "Tweet", "label": [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]}

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)

        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                if line_idx == 0:
                    continue
                fields = line.split("\t")
                text = fields[1]
                ref_labels = []
                for j in range(2, 13):
                    ref_labels.append(int(fields[j]))
                data.append(
                    {"input": text, "label": ref_labels, "line_number": line_idx}
                )

        return data
