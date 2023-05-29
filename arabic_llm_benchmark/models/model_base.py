from abc import ABC, abstractmethod

from pathlib import Path

import traceback
import sys

class ModelBase(object):
	def __init__(self, max_retries=3, **kwargs):
		self.max_retries = max_retries

	@abstractmethod
	def prompt(self, **kwargs):
		pass

	def run_model(self, **kwargs):
		for try_idx in range(self.max_retries):
			try:
				response = self.prompt(**kwargs)
				return {"response": response}
			except Exception as e:
				exc_info = sys.exc_info()
				exception_str = ''.join(traceback.format_exception(*exc_info))

		return {"failure_exception": exception_str}
