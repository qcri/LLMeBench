import csv

from pathlib import Path

from llmebench.datasets.dataset_base import DatasetBase


class CSVDataset(DatasetBase):
    """
    Generic CSV dataset loader

    This data loader provides a way to load local csv/tsv datasets from disk. Assets
    using this loader *must* provide a `custom_test_split`, which can be a relative
    path which will be resolved relative to `data_dir`, or an absolute path. Similarly,
    `custom_train_split` must also be provided for few shot assets.

    Attributes
    ----------
    data_dir : str
        Base path of data containing all datasets. Defaults to "data" in the current
        working directory.
    column_mapping : dict
        Mapping defining which of the columns in the loaded csv are "input" and "label".
        The supplied dict must contain mappings for "input" and "label", and may contain
        other mappings (such as "input_id"). Column mappings can be `int`'s, which would
        be used as indices, or `str`'s, which would be used to search for column indices
        in a header row
    has_header : bool, defaults to True
        Whether the file has a header. If column_mapping specifies column names as `str`,
        this must be True. Defaults to True.
    delimiter : str, defaults to ','
        Delimiter for the csvreader
    encoding : str, defaults to 'utf=8'
        Encoding to use when opening the file
    """

    def __init__(
        self, column_mapping, has_header=True, delimiter=",", encoding="utf-8", **kwargs
    ):
        # Check for column_mapping
        assert "input" in column_mapping
        assert "label" in column_mapping
        self.column_mapping = column_mapping

        self.has_header = has_header
        self.delimiter = delimiter
        self.encoding = encoding

        super(CSVDataset, self).__init__(**kwargs)

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

        with open(data_path, "r", encoding=self.encoding) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=self.delimiter)

            header = None
            if self.has_header:
                header = next(csv_reader)

            column_index_mapping = {}
            for sample_key, column_ref in self.column_mapping.items():
                if isinstance(column_ref, int):
                    column_index_mapping[sample_key] = column_ref
                elif isinstance(column_ref, str):
                    assert (
                        header is not None
                    ), f"CSV Loader: file must have header if column_mapping uses `str` values"
                    column_idx = header.index(column_ref)
                    assert (
                        column_idx != -1
                    ), f"CSV Loader: {column_ref} not found in data"
                    column_index_mapping[sample_key] = column_idx
                else:
                    raise Exception(
                        f"CSV Loader: column_mapping must use `int` or `str` values"
                    )

            for row in csv_reader:
                processed_sample = {}
                for sample_key, column_idx in column_index_mapping.items():
                    processed_sample[sample_key] = row[column_idx]
                data.append(processed_sample)

        return data

    @classmethod
    def download_dataset(cls, data_dir, download_url=None, default_url=None):
        # Generic dataset loaders do not refer to a specific dataset to download
        pass
