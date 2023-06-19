from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class EmotionDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(EmotionDataset, self).__init__(**kwargs)

    def citation(self):
        return """@misc{hassan2022crosslingual,
                title={Cross-lingual Emotion Detection}, 
                author={Sabit Hassan and Shaden Shaar and Kareem Darwish},
                year={2022},
                eprint={2106.06017},
                archivePrefix={arXiv},
                primaryClass={cs.CL}
                }"""

    def get_data_sample(self):
        return {"input": "Tweet", "label": [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]}

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                if line_idx == 0:
                    continue
                fields = line.split("\t")
                text = fields[1]
                ref_labels = []
                for j in range(2, 13):
                    ref_labels.append(int(fields[j]))
                data.append(
                    {"input": text, "label": ref_labels, "line_number": line_idx}
                )

        return data
        
