from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class ArapTweetDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ArapTweetDataset, self).__init__(**kwargs)

    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{zaghouani2018arap,
                title={Arap-Tweet: A Large Multi-Dialect Twitter Corpus for Gender, Age and Language Variety Identification},
                author={Zaghouani, Wajdi and Charfi, Anis},
                booktitle={Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
                year={2018}
            }""",
            "splits": {
                "test": "data/demographic_attributes/gender/test-ARAP-unique.txt",
                "train": "data/demographic_attributes/gender/train-wajdi.tsv",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["Female", "Male"],
        }

    def get_data_sample(self):
        return {"input": "A name", "label": "m"}

    def load_data(self, data_path, no_labels=False):
        data = []
        if "test" in data_path:
            with open(data_path, "r") as fp:
                for line_idx, line in enumerate(fp):
                    name, label = line.strip().split("\t")
                    data.append(
                        {
                            "input": name,
                            "input_id": name,
                            "label": label,
                            "line_number": line_idx,
                        }
                    )
        else:
            user_ids = set()
            with open(data_path, "r") as fp:
                for line_idx, line in enumerate(fp):
                    line = line.strip()

                    # Ignore empty lines
                    if len(line) == 0:
                        continue

                    arr = line.split("\t")

                    # Ignore lines that do not have all columns
                    if len(arr) != 6:
                        continue

                    # Do not add the same user twice
                    user_id = arr[0].strip()
                    if user_id in user_ids:
                        continue
                    user_ids.add(user_id)

                    # Set `input_id` to name to deduplicate based on same name
                    name = arr[1].strip()
                    label = arr[3].strip()
                    data.append(
                        {
                            "input": name,
                            "label": label,
                            "input_id": user_id,
                            "line_number": line_idx,
                        }
                    )

        return data
