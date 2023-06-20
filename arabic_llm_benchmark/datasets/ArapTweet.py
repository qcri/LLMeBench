from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class ArapTweetDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ArapTweetDataset, self).__init__(**kwargs)

    def citation(self):
        return """@inproceedings{zaghouani2018arap,
              title={Arap-Tweet: A Large Multi-Dialect Twitter Corpus for Gender, Age and Language Variety Identification},
              author={Zaghouani, Wajdi and Charfi, Anis},
              booktitle={Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
              year={2018}
            }
        """

    def get_data_sample(self):
        return {"input": "A name", "label": "m"}

    def load_data(self, data_path, no_labels=False):
        data = []
        if "test" in data_path:
            with open(data_path, "r") as fp:
                for line_idx, line in enumerate(fp):
                    name, label = line.strip().split("\t")
                    data.append(
                        {"input": name, "label": label, "line_number": line_idx}
                    )
        else:
            with open(data_path, "r") as fp:
                for line_idx, line in enumerate(fp):
                    arr = line.strip().split("\t")
                    id = arr[0]
                    name = arr[1]
                    label = arr[3]
                    data.append(
                        {
                            "input": name,
                            "label": label,
                            "input_id": id,
                            "line_number": line_idx,
                        }
                    )

        return data
