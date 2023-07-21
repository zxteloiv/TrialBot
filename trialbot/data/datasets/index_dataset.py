from trialbot.data.dataset import Dataset


class IndexDataset(Dataset):
    def __init__(self, dataset_store: Dataset, index: list[int]):
        self._store = dataset_store
        self.index = index

    def __len__(self):
        return len(self.index)

    def get_example(self, i: int):
        return self._store[self.index[i]]