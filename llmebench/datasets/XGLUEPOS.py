from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class XGLUEPOSDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(XGLUEPOSDataset, self).__init__(**kwargs)

    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{liang2020xglue,
                title={XGLUE: A new benchmark datasetfor cross-lingual pre-training, understanding and generation},
                author={Liang, Yaobo and Duan, Nan and Gong, Yeyun and Wu, Ning and Guo, Fenfei and Qi, Weizhen and Gong, Ming and Shou, Linjun and Jiang, Daxin and Cao, Guihong and others},
                booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
                pages={6008--6018},
                year={2020}
            }""",
            "link": "https://microsoft.github.io/XGLUE/",
            "license": "Non-commercial research purposes only",
            "splits": {
                "dev": "data/sequence_tagging_ner_pos_etc/POS/XGLUE/ar.dev.src-trg.txt",
                "test": "data/sequence_tagging_ner_pos_etc/POS/XGLUE/ar.test.src-trg.txt",
            },
            "task_type": TaskType.Labeling,
            "class_labels": [],
        }

    def get_data_sample(self):
        return {
            "input": "Original sentence",
            "label": "Sentence with POS tags",
        }

    def load_data(self, data_path, no_labels=False):
        data = []

        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                data.append(
                    {
                        "input": line.strip().split("\t")[0],
                        "label": line.strip().split("\t")[1],
                        "line_number": line_idx,
                    }
                )

        return data
