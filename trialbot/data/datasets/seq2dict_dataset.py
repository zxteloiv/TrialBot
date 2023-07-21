from ..dataset import Dataset, CompositionalDataset


class SequentialToDict(CompositionalDataset):
    def __init__(self, dataset: Dataset, keys: list[str]):
        super().__init__(dataset)
        self.keys = keys

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i: int):
        iterable_data = self.dataset.get_example(i)
        return dict(zip(self.keys, iterable_data))
