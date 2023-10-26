import datasets

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class HuggingFaceDataset(DatasetBase):
    """
    Generic HuggingFace dataset loader

    This data loader provides a way to load datasets on HuggingFace Hub and transform
    them into the format required by the framework.

    Attributes
    ----------
    data_dir : str
        Base path of data containing all datasets. Defaults to "data" in the current
        working directory.
    huggingface_dataset_name : str
        Name of the dataset on HuggingFace Hub (e.g. 'sst2')
    column_mapping : dict
        Mapping defining which of the columns in the loaded data are "input" and "label".
        The supplied dict must contain mappings for "input" and "label", and may contain
        other mappings (such as "input_id").
    """

    def __init__(self, huggingface_dataset_name, column_mapping, **kwargs):
        self.huggingface_dataset_name = huggingface_dataset_name

        # Check for column_mapping
        assert "input" in column_mapping
        assert "label" in column_mapping
        self.column_mapping = column_mapping

        # Loading the dataset will automatically download it
        self.dataset = datasets.load_dataset(
            huggingface_dataset_name, cache_dir=kwargs["data_dir"]
        )

        super(HuggingFaceDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "multilingual",
            "citation": """@inproceedings{Lhoest_Datasets_A_Community_2021,
                author = {Lhoest, Quentin and Villanova del Moral, Albert and von Platen, Patrick and Wolf, Thomas and Šaško, Mario and Jernite, Yacine and Thakur, Abhishek and Tunstall, Lewis and Patil, Suraj and Drame, Mariama and Chaumond, Julien and Plu, Julien and Davison, Joe and Brandeis, Simon and Sanh, Victor and Le Scao, Teven and Canwen Xu, Kevin and Patry, Nicolas and Liu, Steven and McMillan-Major, Angelina and Schmid, Philipp and Gugger, Sylvain and Raw, Nathan and Lesage, Sylvain and Lozhkov, Anton and Carrigan, Matthew and Matussière, Théo and von Werra, Leandro and Debut, Lysandre and Bekman, Stas and Delangue, Clément},
                booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
                month = nov,
                pages = {175--184},
                publisher = {Association for Computational Linguistics},
                title = {{Datasets: A Community Library for Natural Language Processing}},
                url = {https://aclanthology.org/2021.emnlp-demo.21},
                year = {2021}
            }""",
            "link": "https://huggingface.co/datasets/",
            "license": "Mixed",
            "splits": {
                "None. HuggingFaceDataset asset must use custom_test_split and custom_train_split": None,
            },
            "task_type": TaskType.Other,
        }

    @staticmethod
    def get_data_sample():
        return {"input": "Test Input", "label": "0"}

    def load_data(self, data_split, no_labels=False):
        data = []
        for sample in self.dataset[data_split]:
            processed_sample = {}
            for sample_key, column_name in self.column_mapping.items():
                processed_sample[sample_key] = sample[column_name]
            data.append(processed_sample)

        return data

    @classmethod
    def download_dataset(cls, data_dir, download_url=None, default_url=None):
        # HuggingFace datasets library has its own data downloader
        pass
