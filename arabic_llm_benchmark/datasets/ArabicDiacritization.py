from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class ArabicDiacritizationDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ArabicDiacritizationDataset, self).__init__(**kwargs)

    def citation(self):
        return """@article{10.1145/3434235,
            author = {Darwish, Kareem and Abdelali, Ahmed and Mubarak, Hamdy and Eldesouki, Mohamed},
            title = {Arabic Diacritic Recovery Using a Feature-Rich BiLSTM Model},
            year = {2021},
            issue_date = {March 2021},
            publisher = {Association for Computing Machinery},
            address = {New York, NY, USA},
            volume = {20},
            number = {2},
            issn = {2375-4699},
            url = {https://doi.org/10.1145/3434235},
            doi = {10.1145/3434235},
            journal = {ACM Trans. Asian Low-Resour. Lang. Inf. Process.},
            month = {apr},
            articleno = {33},
            numpages = {18},
            }"""

    def get_data_sample(self):
        return {
            "input": "Original sentence",
            "label": "Sentence with diacritized words",
        }

    def load_data(self, data_path, no_labels=False):
        data = []

        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                text, diacritized_text = line.split("\t")
                data.append(
                    {
                        "input": text.strip(),
                        "label": diacritized_text.strip(),
                        "line_number": line_idx,
                    }
                )

        return data
