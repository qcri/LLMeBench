from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class Khouja20StanceDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(Khouja20StanceDataset, self).__init__(**kwargs)

    def citation(self):
        return """
        @article{khouja2020stance,
            title={Stance prediction and claim verification: An Arabic perspective},
            author={Khouja, Jude},
            journal={arXiv preprint arXiv:2005.10410},
            year={2020}
        }"""

    def get_data_sample(self):
        return {
            "input": {
                "sentence_1": "Sentence in language #1",
                "sentence_2": "Sentence in language #2",
            },
            "label": "agree",
        }

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
        data = []
        with open(data_path, "r") as fp:
            next(fp)  # skip header
            for line_idx, line in enumerate(fp):
                s1, s2, label = line.strip().split(",")
                data.append(
                    {"input": {"sentence_1": s1, "sentence_2": s2}, "label": label}
                )

        return data
