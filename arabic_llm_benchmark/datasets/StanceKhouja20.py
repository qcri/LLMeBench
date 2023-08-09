from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class StanceKhouja20Dataset(DatasetBase):
    def __init__(self, **kwargs):
        super(StanceKhouja20Dataset, self).__init__(**kwargs)

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
                "sentence_1": "الجملة الاولى",
                "sentence_2": "الجملة الثانية",
            },
            "label": "agree",
        }

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
        data = []
        with open(data_path, "r", encoding="utf-8") as fp:
            next(fp)  # skip header
            for line_idx, line in enumerate(fp):
                s1, s2, label = line.strip().split(",")

                # Had to concatenate s1 and s2 this as langchain only accepts strings
                data.append(
                    {
                        "input": {"sentence_1": s1.strip(), "sentence_2": s2.strip()},
                        "label": label,
                        "line_number": line_idx,
                    }
                )

        return data
