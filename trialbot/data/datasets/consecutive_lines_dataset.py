from trialbot.data.datasets.file_dataset import FileDataset

class ConsecutiveLinesDataset(FileDataset):
    def __init__(self, filename, lazy=True, num_lines=2):
        super(ConsecutiveLinesDataset, self).__init__(filename, lazy=lazy)
        self._num_lines = num_lines

    def __len__(self):
        return super().__len__() // 2

    def get_example(self, i):
        example = tuple(
            super().get_example(idx)
            for idx in range(i * self._num_lines, (i + 1) * self._num_lines)
        )
        return example
