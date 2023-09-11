import http.server
import threading
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from llmebench.datasets.dataset_base import DatasetBase


class ArchiveHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, directory="tests/datasets/archives")


class SignalingHTTPServer(http.server.HTTPServer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ready_event = threading.Event()

    def service_actions(self):
        self.ready_event.set()


class MockDataset(DatasetBase):
    def metadata(self):
        return {}

    def get_data_sample(self):
        return {"input": "input", "label": "label"}

    def load_data(self, data_path):
        return [self.get_data_sample() for _ in range(100)]


class TestDatasetAutoDownload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = SignalingHTTPServer(("", 8076), ArchiveHandler)
        cls.test_server = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.test_server.start()
        cls.httpd.ready_event.wait()

    @classmethod
    def tearDownClass(cls):
        if cls.httpd:
            cls.httpd.shutdown()
            cls.httpd.server_close()
        cls.test_server.join()

    def test_auto_download_zip(self):
        "Test automatic downloading and extraction of *.zip datasets"

        data_dir = TemporaryDirectory()

        dataset = MockDataset(data_dir=data_dir.name)
        self.assertTrue(
            dataset.download_dataset(
                download_url="http://localhost:8076/MockDataset.zip"
            )
        )

        downloaded_files = list(Path(data_dir.name).iterdir())
        downloaded_filenames = [f.name for f in downloaded_files if f.is_file()]
        self.assertEqual(len(downloaded_files), 2)
        self.assertIn("MockDataset.zip", downloaded_filenames)

        extracted_directories = [d for d in downloaded_files if d.is_dir()]
        extracted_directory_names = [d.name for d in extracted_directories]
        self.assertIn("MockDataset", extracted_directory_names)
        self.assertEqual(len(extracted_directory_names), 1)

        dataset_files = [f.name for f in extracted_directories[0].iterdir()]
        self.assertIn("train.txt", dataset_files)
        self.assertIn("test.txt", dataset_files)

    def test_auto_download_tar(self):
        "Test automatic downloading and extraction of *.tar datasets"

        data_dir = TemporaryDirectory()

        dataset = MockDataset(data_dir=data_dir.name)
        self.assertTrue(
            dataset.download_dataset(
                download_url="http://localhost:8076/MockDataset.tar"
            )
        )

        downloaded_files = list(Path(data_dir.name).iterdir())
        downloaded_filenames = [f.name for f in downloaded_files if f.is_file()]
        self.assertEqual(len(downloaded_files), 2)
        self.assertIn("MockDataset.tar", downloaded_filenames)

        extracted_directories = [d for d in downloaded_files if d.is_dir()]
        extracted_directory_names = [d.name for d in extracted_directories]
        self.assertIn("MockDataset", extracted_directory_names)
        self.assertEqual(len(extracted_directory_names), 1)

        dataset_files = [f.name for f in extracted_directories[0].iterdir()]
        self.assertIn("train.txt", dataset_files)
        self.assertIn("test.txt", dataset_files)

    def test_auto_download_tar_gz(self):
        "Test automatic downloading and extraction of *.tar.gz datasets"

        data_dir = TemporaryDirectory()

        dataset = MockDataset(data_dir=data_dir.name)
        self.assertTrue(
            dataset.download_dataset(
                download_url="http://localhost:8076/MockDataset.tar.gz"
            )
        )

        downloaded_files = list(Path(data_dir.name).iterdir())
        self.assertEqual(len(downloaded_files), 2)
        downloaded_filenames = [f.name for f in downloaded_files if f.is_file()]
        self.assertIn("MockDataset.tar.gz", downloaded_filenames)

        extracted_directories = [d for d in downloaded_files if d.is_dir()]
        extracted_directory_names = [d.name for d in extracted_directories]
        self.assertIn("MockDataset", extracted_directory_names)
        self.assertEqual(len(extracted_directory_names), 1)

        dataset_files = [f.name for f in extracted_directories[0].iterdir()]
        self.assertIn("train.txt", dataset_files)
        self.assertIn("test.txt", dataset_files)

    @patch.dict(
        "os.environ",
        {
            "DEFAULT_DOWNLOAD_URL": "http://localhost:8076/",
        },
    )
    def test_auto_download_default_url(self):
        "Test automatic downloading when download url is not provided"

        data_dir = TemporaryDirectory()

        dataset = MockDataset(data_dir=data_dir.name)
        self.assertTrue(dataset.download_dataset())

        downloaded_files = list(Path(data_dir.name).iterdir())
        downloaded_filenames = [f.name for f in downloaded_files if f.is_file()]
        self.assertEqual(len(downloaded_files), 2)
        self.assertIn("MockDataset.zip", downloaded_filenames)

        extracted_directories = [d for d in downloaded_files if d.is_dir()]
        extracted_directory_names = [d.name for d in extracted_directories]
        self.assertIn("MockDataset", extracted_directory_names)
        self.assertEqual(len(extracted_directory_names), 1)

        dataset_files = [f.name for f in extracted_directories[0].iterdir()]
        self.assertIn("train.txt", dataset_files)
        self.assertIn("test.txt", dataset_files)


class TestDatasetCaching(unittest.TestCase):
    def test_cache_existing_file(self):
        "Test if an existing file _does not_ trigger a download"

        data_dir = TemporaryDirectory()
        archive_file = Path("tests/datasets/archives/MockDataset.zip")
        copy_archive_file = Path(data_dir.name) / "MockDataset.zip"
        copy_archive_file.write_bytes(archive_file.read_bytes())

        dataset = MockDataset(data_dir=data_dir.name)
        self.assertTrue(
            dataset.download_dataset(
                download_url="http://localhost:8076/ExistingData.zip"
            )
        )
