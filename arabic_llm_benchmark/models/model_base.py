from abc import ABC, abstractmethod

from pathlib import Path
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

import traceback
import sys

def log_retry(retry_state):
	if retry_state.attempt_number == 1:
		return
	print(f"Request failed, retry attempt {retry_state.attempt_number}...")

class ModelBase(object):
	def __init__(self, max_retries=2, retry_exceptions=(), **kwargs):
		self.max_retries = max_retries

		# Instantiate retrying mechanism
		self.prompt = retry(
			wait=wait_random_exponential(multiplier=1, max=60),
			stop=stop_after_attempt(self.max_retries),
			retry=retry_if_exception_type(retry_exceptions),
			before=log_retry,
			reraise=True
		)(self.prompt)

	@abstractmethod
	def prompt(self, **kwargs):
		pass

	def run_model(self, **kwargs):
		try:
			response = self.prompt(**kwargs)
			return {"response": response}
		except Exception as e:
			exc_info = sys.exc_info()
			exception_str = ''.join(traceback.format_exception(*exc_info))

		return {"failure_exception": exception_str}
