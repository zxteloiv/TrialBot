from ..dataset import Dataset

import trialbot.utils.file_reader as reader_utils

class FileDataset(Dataset):
    def __init__(self, filename, lazy=True):
        super(FileDataset, self).__init__()
        self.lazy = lazy

        self.filename = filename

        self._data = None
        if not lazy:
            self._read_data()

    def _read_data(self):
        if self._data is None:
            self._data = list(line.rstrip('\r\n') for line in reader_utils.open_file(self.filename))

    def __len__(self):
        self._read_data()
        return len(self._data)

    def get_example(self, i):
        self._read_data()
        return self._data[i]

class DirectoryDataset(Dataset):
    def __init__(self, dirname):
        super(DirectoryDataset, self).__init__()
        self.dirname = dirname
        self._data = list(line.rstrip('\r\n') for line in reader_utils.open_dir(dirname))

    def __len__(self):
        return len(self._data)

    def get_example(self, i):
        return self._data[i]

