from abc import ABC, abstractmethod

from pathlib import Path

import json

class ModelBase(object):
	def __init__(self, cache_dir, **kwargs):
		self.cache_dir = Path(cache_dir)

		if not self.cache_dir.exists():
			self.cache_dir.mkdir(parents=True)

	@abstractmethod	
	def prompt(self, **kwargs):
		pass

	def run_model(self, sample_key, **kwargs):
		cache_path = self.cache_dir / f"{sample_key}.json"

		if cache_path.exists():
			with open(cache_path, "r") as fp:	
				response = json.load(fp)
		else:
			# TODO: save abnormal responses as well
			response = self.prompt(**kwargs)

			# Save raw response to file
			# TODO: Save inputs as well
			with open(cache_path, "w") as fp:
				json.dump(response, fp)

		return response
