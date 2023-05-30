import inspect
import unittest

from glob import glob
from pathlib import Path

import arabic_llm_benchmark.tasks as tasks
from arabic_llm_benchmark import Benchmark, utils


class TestTaskExports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Search for all implemented tasks
        framework_dir = Path("arabic_llm_benchmark")
        cls.implemented_tasks = [
            task_path
            for task_path in glob(str(framework_dir / "tasks" / "*.py"))
            if "__init__" not in task_path and "task_base" not in task_path
        ]
        cls.exported_tasks = set(
            [m[1].__name__ for m in inspect.getmembers(tasks, inspect.isclass)]
        )

    def test_task_exports(self):
        "Test if all implemented tasks are exported in __init__.py"

        for task_path in self.implemented_tasks:
            with self.subTest(msg=task_path):
                implemented_module = utils.import_source_file(
                    Path(task_path), "test_module"
                )
                implemented_class = [
                    c
                    for c in inspect.getmembers(implemented_module, inspect.isclass)
                    if c[0] != "TaskBase"
                ][0]
                implemented_task = implemented_class[1].__name__

                self.assertIn(
                    implemented_task,
                    self.exported_tasks,
                    msg=f"{implemented_task} not exported in tasks/__init__.py",
                )
