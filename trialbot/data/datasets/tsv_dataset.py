from trialbot.data.datasets.file_dataset import FileDataset

class TSVDataset(FileDataset):
    def __init__(self, filename, sep="\t", has_title=True, lazy=True):
        super().__init__(filename, lazy)
        self._sep = sep
        self._has_title = has_title
        self._fields = None
        if (not self.lazy) and len(self._data) > 0:
            self._fields = self._data[0].split(self._sep)

    def get_example(self, i):
        example = super().get_example(i + 1)
        example_tuple = example.split(self._sep)
        if not self._has_title:
            return example_tuple

        if self._fields is None:
            self._fields = self._data[0].split(self._sep)

        example_dict = dict(zip(self._fields, example_tuple))
        return example_dict

    def __len__(self):
        length = super().__len__()
        if self._has_title:
            length -= 1
        return length
