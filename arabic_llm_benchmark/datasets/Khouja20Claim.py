from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class Khouja20ClaimDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(Khouja20ClaimDataset, self).__init__(**kwargs)

    def citation(self):
        return """
        @article{khouja2020stance,
            title={Stance prediction and claim verification: An Arabic perspective},
            author={Khouja, Jude},
            journal={arXiv preprint arXiv:2005.10410},
            year={2020}
        }"""

    def get_data_sample(self):
        return {"input": "الجملة بالعربية", "label": "yes"}

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
        data = []
        with open(data_path, "r") as fp:
            next(fp)  # skip the header
            for line in fp:
                claim_s, fake_flag = line.strip().split(",")
                data.append({"input": fake_flag, "label": claim_s})

        return data
