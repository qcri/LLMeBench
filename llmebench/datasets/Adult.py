from llmebench.datasets.dataset_base import DatasetBase


class AdultDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(AdultDataset, self).__init__(**kwargs)

    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{mubarak-etal-2021-adult,
                title = "Adult Content Detection on {A}rabic {T}witter: Analysis and Experiments",
                author = "Mubarak, Hamdy  and
                  Hassan, Sabit  and
                  Abdelali, Ahmed",
                booktitle = "Proceedings of the Sixth Arabic Natural Language Processing Workshop",
                month = apr,
                year = "2021",
                address = "Kyiv, Ukraine (Virtual)",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/2021.wanlp-1.14",
                pages = "136--144",
            }""",
        }

    def get_data_sample(self):
        return {"input": "نص عادي", "label": "NOT_ADULT"}

    def load_data(self, data_path, no_labels=False):
        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                fields = line.split("\t")
                label = fields[0]
                text = fields[4]
                input_id = fields[7]
                data.append(
                    {
                        "input": text,
                        "label": label,
                        "input_id": input_id,
                        "line_number": line_idx,
                    }
                )

        return data
