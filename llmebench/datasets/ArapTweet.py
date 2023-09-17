from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class ArapTweetDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ArapTweetDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{zaghouani2018arap,
                title={Arap-Tweet: A Large Multi-Dialect Twitter Corpus for Gender, Age and Language Variety Identification},
                author={Zaghouani, Wajdi and Charfi, Anis},
                booktitle={Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
                year={2018}
            }
            @inproceedings{zaghouani2018guidelines,
              title={Guidelines and Annotation Framework for Arabic Author Profiling},
              author={Zaghouani, Wajdi and Charfi, Anis},
              booktitle={OSACT 3: The 3rd Workshop on Open-Source Arabic Corpora and Processing Tools},
              pages={68},
              year={2018}
            }
            @inproceedings{charfi2019fine,
              title={A Fine-Grained Annotated Multi-Dialectal Arabic Corpus},
              author={Charfi, Anis and Zaghouani, Wajdi and Mehdi, Syed Hassan and Mohamed, Esraa},
              booktitle={Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2019)},
              pages={198--204},
              year={2019}
            }""",
            "splits": {
                "test": "test.tsv",
                "train": "train.tsv",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["Female", "Male"],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "A name", "label": "m"}

    def load_data(self, data_path, no_labels=False):
        data = []
        if "test" in data_path:
            data_path = self.resolve_path(data_path)
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
            data_path = self.resolve_path(data_path)
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
