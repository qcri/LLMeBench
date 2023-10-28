from llmebench.datasets.CSV import CSVDataset


class TSVDataset(CSVDataset):
    """
    Generic TSV dataset loader

    This data loader provides a way to load local tsv datasets from disk. Assets
    using this loader *must* provide a `custom_test_split`, which can be a relative
    path which will be resolved relative to `data_dir`, or an absolute path. Similarly,
    `custom_train_split` must also be provided for few shot assets.

    Attributes
    ----------
    data_dir : str
        Base path of data containing all datasets. Defaults to "data" in the current
        working directory.
    column_mapping : dict
        Mapping defining which of the columns in the loaded tsv are "input" and "label".
        The supplied dict must contain mappings for "input" and "label", and may contain
        other mappings (such as "input_id"). Column mappings can be `int`'s, which would
        be used as indices, or `str`'s, which would be used to search for column indices
        in a header row
    has_header : bool, defaults to True
        Whether the file has a header. If column_mapping specifies column names as `str`,
        this must be True. Defaults to True.
    encoding : str, defaults to 'utf=8'
        Encoding to use when opening the file
    """

    def __init__(self, **kwargs):
        kwargs["delimiter"] = "\t"

        super(TSVDataset, self).__init__(**kwargs)
