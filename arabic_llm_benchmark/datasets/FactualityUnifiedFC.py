from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class FactualityUnifiedFCDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(FactualityUnifiedFCDataset, self).__init__(**kwargs)

    def citation(self):
        return """
                @inproceedings{baly2018integrating,
                  title = "Integrating Stance Detection and Fact Checking in a Unified Corpus",
                    author = "Baly, Ramy  and
                      Mohtarami, Mitra  and
                      Glass, James  and
                      M{\\`a}rquez, Llu{\\'\\i}s  and
                      Moschitti, Alessandro  and
                      Nakov, Preslav",
                    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)",
                    year = "2018",
                }
        """

    def get_data_sample(self):
        return {"input": "الجملة الاولى", "label": "agree", "input_id": "id"}

    def load_data(self, data_path):
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            header = next(f)
            if (
                "," in header
            ):  # A trick to check if we are loading train data for FS from Khouja 20
                for line_idx, line in enumerate(f):
                    sentence, label_fixed = [str(s.strip()) for s in line.split(",")]

                    # The dataset uses 1 to reflect false/fake claims
                    if label_fixed == "1":
                        label_fixed = "false"
                    elif label_fixed == "0":
                        label_fixed = "true"

                    data.append(
                        {
                            "input": sentence,
                            "label": label_fixed,
                            "line_number": line_idx,
                        }
                    )
            else:  # Load test data from UnifiedFC
                for line_idx, line in enumerate(f):
                    input_id, sentence, label = [
                        str(s.strip()) for s in line.split("\t")
                    ]

                    data.append(
                        {
                            "input": sentence,
                            "label": label,
                            "line_number": line_idx,
                            "input_id": input_id,
                        }
                    )

        return data
