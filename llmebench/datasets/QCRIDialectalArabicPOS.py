from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class QCRIDialectalArabicPOSDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(QCRIDialectalArabicPOSDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@InProceedings{DARWISH18.562,
                author = {Kareem Darwish , Hamdy Mubarak , Ahmed Abdelali , Mohamed Eldesouki , Younes Samih , Randah Alharbi , Mohammed Attia , Walid Magdy and Laura Kallmeyer},
                title = {Multi-Dialect Arabic POS Tagging: A CRF Approach},
                booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
                year = {2018},
                month = {may},
                date = {7-12},
                location = {Miyazaki, Japan},
                editor = {Nicoletta Calzolari (Conference chair) and Khalid Choukri and Christopher Cieri and Thierry Declerck and Sara Goggi and Koiti Hasida and Hitoshi Isahara and Bente Maegaard and Joseph Mariani and Hélène Mazo and Asuncion Moreno and Jan Odijk and Stelios Piperidis and Takenobu Tokunaga},
                publisher = {European Language Resources Association (ELRA)},
                address = {Paris, France},
                isbn = {979-10-95546-00-9},
                language = {english}
            }""",
            "link": "https://alt.qcri.org/resources/da_resources/",
            "license": "Apache License, Version 2.0",
            "splits": {
                "glf.data_5": {
                    "dev": "glf.pos/glf.data_5.dev.src-trg.sent",
                    "test": "glf.pos/glf.data_5.test.src-trg.sent",
                },
                "lev.data_5": {
                    "dev": "lev.pos/lev.data_5.dev.src-trg.sent",
                    "test": "lev.pos/lev.data_5.test.src-trg.sent",
                },
                "egy.data_5": {
                    "dev": "egy.pos/egy.data_5.dev.src-trg.sent",
                    "test": "egy.pos/egy.data_5.test.src-trg.sent",
                },
                "mgr.data_5": {
                    "dev": "mgr.pos/mgr.data_5.dev.src-trg.sent",
                    "test": "mgr.pos/mgr.data_5.test.src-trg.sent",
                },
                "default": ["glf.data_5", "lev.data_5", "egy.data_5", "mgr.data_5"],
            },
            "task_type": TaskType.SequenceLabeling,
            "class_labels": [
                "ADJ",
                "ADV",
                "CASE",
                "CONJ",
                "DET",
                "EMOT",
                "FOREIGN",
                "FUT_PART",
                "HASH",
                "MENTION",
                "NEG_PART",
                "NOUN",
                "NSUFF",
                "NUM",
                "PART",
                "PREP",
                "PROG_PART",
                "PRON",
                "PUNC",
                "URL",
                "V",
            ],
        }

    @staticmethod
    def get_data_sample():
        return {
            "input": "Original sentence",
            "label": "Sentence with POS tags",
        }

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)

        data = []

        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                data.append(
                    {
                        "input": line.strip().split("\t")[0],
                        "label": line.strip().split("\t")[1],
                        "line_number": line_idx,
                    }
                )

        return data
