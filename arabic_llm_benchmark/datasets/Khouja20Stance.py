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

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                claim_s, fake_flag = line.strip().split(",")
                data.append(
                    {"input": fake_flag, "label": claim_s[-1], "line_number": line_idx}
                )

        return data
