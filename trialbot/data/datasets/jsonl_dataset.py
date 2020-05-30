from trialbot.data.datasets.file_dataset import FileDataset
import json

class JsonLDataset(FileDataset):
    def get_example(self, i):
        line = super().get_example(i)
        return json.loads(line)