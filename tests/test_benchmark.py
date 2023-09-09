import types

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from unittest.mock import MagicMock, patch

from llmebench import Benchmark


class MockAsset(object):
    @staticmethod
    def config():
        return {}


@patch("llmebench.utils.import_source_file", MagicMock(return_value=MockAsset))
class TestBenchmarkAssetFinder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets_dir = TemporaryDirectory()
        cls.assets_path = cls.assets_dir.name

        cls.assets = [
            "generic_zeroshot.py",
            "ar/unique_arabic_asset.py",
            "ar/sentiment/sentiment_zeroshot.py",
            "ar/sentiment/sentiment_fewshot.py",
            "en/sentiment/sentiment_zeroshot.py",
            "unique_prefix/task_1/asset_1.py",
            "unique_prefix/task_1/asset_2.py",
            "unique_prefix/task_1/asset_3.py",
            "unique_prefix/task_2/asset_1.py",
        ]

        non_assets = ["sometempfile.txt" "ar/arabic_runner.pyc"]

        for asset in cls.assets + non_assets:
            asset_path = Path(cls.assets_path) / asset
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.touch(exist_ok=True)

        cls.benchmark = Benchmark(benchmark_dir=cls.assets_path)

    @classmethod
    def tearDownClass(cls):
        cls.assets_dir.cleanup()

    def test_default(self):
        "Test if all assets are found when benchmark is run"

        self.assertEqual(len(self.benchmark.find_assets()), len(self.assets))

    def test_filename(self):
        "Test if an asset is found by exact file name"

        assets = self.benchmark.find_assets("unique_arabic_asset.py")

        self.assertEqual(len(assets), 1)
        self.assertIn("unique_arabic_asset", assets[0]["name"])

    def test_filename_root(self):
        "Test if an asset is found by exact file name when in the root of the assets folder"

        assets = self.benchmark.find_assets("generic_zeroshot.py")

        self.assertEqual(len(assets), 1)
        self.assertIn("generic_zeroshot", assets[0]["name"])

    def test_wildcard(self):
        "Test if assets are found with wildcard search"

        assets = self.benchmark.find_assets("*zeroshot*")

        self.assertEqual(len(assets), 3)
        for asset in assets:
            self.assertIn("zeroshot", asset["name"])

    def test_full_absolute_path(self):
        "Test if assets are found with their full absolute path"

        assets = self.benchmark.find_assets(
            self.assets_path + "/ar/sentiment/sentiment_zeroshot.py"
        )

        self.assertEqual(len(assets), 1)
        self.assertIn("sentiment_zeroshot", assets[0]["name"])

    def test_full_relative_path(self):
        "Test if assets are found with their full relative path"

        assets = self.benchmark.find_assets("ar/sentiment/sentiment_zeroshot.py")

        self.assertEqual(len(assets), 1)
        self.assertIn("sentiment_zeroshot", assets[0]["name"])

    def test_partial_path(self):
        "Test if assets are found with their partial paths"

        assets = self.benchmark.find_assets("unique_prefix/*")

        self.assertEqual(len(assets), 4)
        for asset in assets:
            self.assertIn("unique_prefix/", asset["name"])
