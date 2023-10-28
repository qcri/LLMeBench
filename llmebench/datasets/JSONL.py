import json

from pathlib import Path

from llmebench.datasets.dataset_base import DatasetBase


class JSONLDataset(DatasetBase):
    """
    Generic jsonl dataset loader

    This data loader provides a way to load local jsonl datasets from disk. Each line
    of the jsonl file must be a valid json object.
    Assets using this loader *must* provide a `custom_test_split`, which can be a
    relative path which will be resolved relative to `data_dir`, or an absolute path.
    Similarly, `custom_train_split` must also be provided for few shot assets.

    Attributes
    ----------
    data_dir : str
        Base path of data containing all datasets. Defaults to "data" in the current
        working directory.
    column_mapping : dict
        Mapping defining which of the keys in the loaded json are "input" and "label".
        The supplied dict must contain mappings for "input" and "label", and may contain
        other mappings (such as "input_id").
    """

    def __init__(self, column_mapping, **kwargs):
        # Check for column_mapping
        assert "input" in column_mapping
        assert "label" in column_mapping
        self.column_mapping = column_mapping

        super(JSONLDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {"generic": True}

    @staticmethod
    def get_data_sample():
        return {"input": "Test Input", "label": "0"}

    def load_data(self, data_split, no_labels=False):
        if not isinstance(data_split, Path):
            data_split = Path(data_split)

        if not data_split.is_absolute():
            data_split = f":data_dir:{data_split}"

        data_path = self.resolve_path(data_split)

        data = []

        with open(data_path, "r") as jsonl_file:
            for line in jsonl_file:
                sample = json.loads(line)

                processed_sample = {}
                for sample_key, column_key in self.column_mapping.items():
                    processed_sample[sample_key] = sample[column_key]
                data.append(processed_sample)

        return data

    @classmethod
    def download_dataset(cls, data_dir, download_url=None, default_url=None):
        # Generic dataset loaders do not refer to a specific dataset to download
        pass
