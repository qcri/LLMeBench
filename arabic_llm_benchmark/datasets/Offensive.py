from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class OffensiveDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(OffensiveDataset, self).__init__(**kwargs)

    def citation(self):
        return """XXX"""

    def get_data_sample(self):
        return {"input": "@USER يلا يا خوخة يا مهزئة ع دراستك", "label": "OFF"}

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
        #Format: text \t offensive_label
        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                text, label = line.split("\t")
                label = label.strip()
                data.append(
                    {"input": text, "label": label, "line_number": line_idx}
                )

        return data
