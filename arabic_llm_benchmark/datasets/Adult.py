from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class AdultDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(AdultDataset, self).__init__(**kwargs)

    def citation(self):
        return """@article{zampieri2020semeval,
          title={SemEval-2020 task 12: Multilingual offensive language identification in social media (OffensEval 2020)},
          author={Zampieri, Marcos and Nakov, Preslav and Rosenthal, Sara and Atanasova, Pepa and Karadzhov, Georgi and Mubarak, Hamdy and Derczynski, Leon and Pitenis, ...},
          journal={arXiv preprint arXiv:2006.07235},
          year={2020}
        }"""

    def get_data_sample(self):
        return {"input": "نص عادي", "label": "NOT_ADULT"}

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
        #Format: text \t adult_label
        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                fields = line.split("\t")
                label = fields[0]
                text = fields[4]                        
                data.append(
                    {"input": text, "label": label, "line_number": line_idx}
                )

        return data
