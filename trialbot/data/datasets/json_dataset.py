import json

from trialbot.data.datasets.file_dataset import FileDataset

class JsonDataset(FileDataset):
    def _read_data(self):
        if self._data is None:
            self._data = json.load(open(self.filename, 'r'))