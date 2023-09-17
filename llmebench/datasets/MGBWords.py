from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class MGBWordsDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(MGBWordsDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{mubarak-etal-2021-qasr,
                title = "{QASR}: {QCRI} Aljazeera Speech Resource A Large Scale Annotated {A}rabic Speech Corpus",
                author = "Mubarak, Hamdy  and
                  Hussein, Amir  and
                  Chowdhury, Shammur Absar  and
                  Ali, Ahmed",
                booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
                month = aug,
                year = "2021",
                address = "Online",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/2021.acl-long.177",
                doi = "10.18653/v1/2021.acl-long.177",
                pages = "2274--2285",
                abstract = "We introduce the largest transcribed Arabic speech corpus, QASR, collected from the broadcast domain. This multi-dialect speech dataset contains 2,000 hours of speech sampled at 16kHz crawled from Aljazeera news channel. The dataset is released with lightly supervised transcriptions, aligned with the audio segments. Unlike previous datasets, QASR contains linguistically motivated segmentation, punctuation, speaker information among others. QASR is suitable for training and evaluating speech recognition systems, acoustics- and/or linguistics- based Arabic dialect identification, punctuation restoration, speaker identification, speaker linking, and potentially other NLP modules for spoken data. In addition to QASR transcription, we release a dataset of 130M words to aid in designing and training a better language model. We show that end-to-end automatic speech recognition trained on QASR reports a competitive word error rate compared to the previous MGB-2 corpus. We report baseline results for downstream natural language processing tasks such as named entity recognition using speech transcript. We also report the first baseline for Arabic punctuation restoration. We make the corpus available for the research community.",
            }""",
            "link": "https://alt.qcri.org/resources/MGB-words.txt",
            "license": "Research Purpose Only",
            "splits": {"test": "MGB-words.txt"},
            "task_type": TaskType.SequenceLabeling,
            "class_labels": [
                "B-PERS",
                "I-PERS",
                "B-LOC",
                "I-LOC",
                "B-ORG",
                "I-ORG",
                "B-MISC",
                "I-MISC",
                "O",
            ],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "sentence", "label": "named entity labels are here"}

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)

        data = []
        with open(data_path, "rt", encoding="UTF8") as fp:
            for idx, line in enumerate(fp):
                if idx == 7906:
                    break
                fields = line.split("\t")
                words = fields[4].split()
                s = ""
                ref_label = []
                for w in words:
                    index1 = w.find("/")
                    if index1 < 0:
                        index1 = len(w)
                        w += "/O"
                    word = w[0:index1]
                    label = w[index1 + 1 :]
                    if label == "B-OTH" or label == "I-OTH":
                        label = "O"
                    ref_label.append(label)

                    s += f"{word} "
                s = s.strip()
                ref_label = " ".join(ref_label)

                data.append({"input": s, "label": ref_label})

        return data
