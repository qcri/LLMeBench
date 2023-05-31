from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class AraBenchDataset(DatasetBase):
    def __init__(self, src, tgt, **kwargs):
        super(AraBenchDataset, self).__init__(**kwargs)
        self.src = src
        self.tgt = tgt

    def citation(self):
        return """@inproceedings{sajjad-etal-2020-arabench,
            title = "{A}ra{B}ench: Benchmarking Dialectal {A}rabic-{E}nglish Machine Translation",
            author = "Sajjad, Hassan  and
              Abdelali, Ahmed  and
              Durrani, Nadir  and
              Dalvi, Fahim",
            booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
            month = dec,
            year = "2020",
            address = "Barcelona, Spain (Online)",
            publisher = "International Committee on Computational Linguistics",
            url = "https://aclanthology.org/2020.coling-main.447",
            doi = "10.18653/v1/2020.coling-main.447",
            pages = "5094--5107"
        }"""

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
        data = []

        with open(data_path + self.src, "r") as fpsrc, open(
            data_path + self.tgt, "r"
        ) as fptgt:
            for line_idx, (srcline, tgtline) in enumerate(zip(fpsrc, fptgt)):
                data.append(
                    {
                        "input": srcline.strip(),
                        "label": tgtline.strip(),
                        "line_number": line_idx,
                    }
                )

        return data
