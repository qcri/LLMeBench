from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class ArabGendDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ArabGendDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{mubarak-etal-2022-arabgend,
                title = "{A}rab{G}end: Gender Analysis and Inference on {A}rabic {T}witter",
                author = "Mubarak, Hamdy  and
                  Chowdhury, Shammur Absar  and
                  Alam, Firoj",
                booktitle = "Proceedings of the Eighth Workshop on Noisy User-generated Text (W-NUT 2022)",
                month = oct,
                year = "2022",
                address = "Gyeongju, Republic of Korea",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/2022.wnut-1.14",
                pages = "124--135",
                abstract = "Gender analysis of Twitter can reveal important socio-cultural differences between male and female users. There has been a significant effort to analyze and automatically infer gender in the past for most widely spoken languages{'} content, however, to our knowledge very limited work has been done for Arabic. In this paper, we perform an extensive analysis of differences between male and female users on the Arabic Twitter-sphere. We study differences in user engagement, topics of interest, and the gender gap in professions. Along with gender analysis, we also propose a method to infer gender by utilizing usernames, profile pictures, tweets, and networks of friends. In order to do so, we manually annotated gender and locations for {\textasciitilde}166K Twitter accounts associated with {\textasciitilde}92K user location, which we plan to make publicly available. Our proposed gender inference method achieve an F1 score of 82.1{\\%} (47.3{\\%} higher than majority baseline). We also developed a demo and made it publicly available.",
            }""",
            "license": "Research Purpose Only",
            "splits": {"test": "gender-test.txt"},
            "task_type": TaskType.Classification,
            "class_labels": ["m", "f"],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "A name", "label": "m"}

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)

        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                label, name = line.strip().split("\t")
                data.append(
                    {"input": name, "label": label[-1], "line_number": line_idx}
                )

        return data
