from abc import ABC, abstractmethod

from pathlib import Path

import json
import traceback
import sys

class ModelBase(object):
	def __init__(self, cache_dir, ignore_cache=False, max_retries=3, **kwargs):
		self.cache_dir = Path(cache_dir)
		self.max_retries = max_retries
		self.ignore_cache = ignore_cache

		if not self.cache_dir.exists():
			self.cache_dir.mkdir(parents=True)

	@abstractmethod
	def prompt(self, **kwargs):
		pass

	def run_model(self, sample_key, **kwargs):
		cache_path = self.cache_dir / f"{sample_key}.json"

		response = None
		if cache_path.exists() and not self.ignore_cache:
			with open(cache_path, "r") as fp:	
				cache_payload = json.load(fp)
				response = cache_payload["response"]
		else:
			for try_idx in range(self.max_retries):
				try:
					response = self.prompt(**kwargs)
					# Save raw response to file
					with open(cache_path, "w") as fp:
						json.dump({
							"input": {
								**kwargs
							},
							"response": response
						}, fp)
					break
				except Exception as e:
					exc_info = sys.exc_info()
					exception_str = ''.join(traceback.format_exception(*exc_info))
					failed_cache_path = self.cache_dir / f"{sample_key}_failed_try_{try_idx+1}.json"
					with open(failed_cache_path, "w") as fp:
						json.dump({
							"input": {
								**kwargs
							},
							"failure_exception": exception_str
						}, fp)

		return response
