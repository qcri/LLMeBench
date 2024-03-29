import inspect
import unittest

from glob import glob
from pathlib import Path

import llmebench.tasks as tasks
from llmebench import Benchmark, utils
from llmebench.tasks.task_base import TaskBase


class TestTaskExports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Search for all implemented tasks
        framework_dir = Path("llmebench")
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
                    if issubclass(c[1], TaskBase) and c[1] != TaskBase
                ][0]
                implemented_task = implemented_class[1].__name__

                self.assertIn(
                    implemented_task,
                    self.exported_tasks,
                    msg=f"{implemented_task} not exported in tasks/__init__.py",
                )
