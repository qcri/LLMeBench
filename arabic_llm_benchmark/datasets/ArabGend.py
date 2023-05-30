from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class ArabGendDataset(DatasetBase):
    def __init__(self):
        pass

    def citation(self):
        return """@article{mubarak2022arabgend,
          title={ArabGend: Gender analysis and inference on {A}rabic Twitter},
          author={Mubarak, Hamdy and Chowdhury, Shammur Absar and Alam, Firoj},
          journal={arXiv preprint arXiv:2203.00271},
          year={2022}
        }"""

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                label, name = line.strip().split("\t")
                data.append(
                    {"input": name, "label": label[-1], "line_number": line_idx}
                )

        return data
