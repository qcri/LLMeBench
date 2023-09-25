from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class WikiNewsLemmatizationDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(WikiNewsLemmatizationDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{mubarak-2018-build,
                title = "Build Fast and Accurate Lemmatization for {A}rabic",
                author = "Mubarak, Hamdy",
                booktitle = "Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)",
                month = may,
                year = "2018",
                address = "Miyazaki, Japan",
                publisher = "European Language Resources Association (ELRA)",
                url = "https://aclanthology.org/L18-1181",
            }""",
            "link": "http://alt.qcri.org/~hmubarak/WikiNews-26-06-2015-RefLemma.xlsx",
            "license": "Research Purpose Only",
            "splits": {"test": "WikiNews-26-06-2015-RefLemma.txt"},
            "task_type": TaskType.Other,
        }

    @staticmethod
    def get_data_sample():
        return {
            "input": "جوائز",
            "label": ("جوائز", "جائزة"),
        }

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)

        # Format: words \t lemmas
        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                words, lemmas = line.split("\t")
                words = words.split()
                lemmas = lemmas.split()
                for i, w in enumerate(words):
                    text = w
                    label = lemmas[i]

                    # Supply origin unlemmatized word in the label as well
                    # to handle failed predictions in evaluate
                    data.append(
                        {"input": text, "label": (text, label), "line_number": line_idx}
                    )

        return data
