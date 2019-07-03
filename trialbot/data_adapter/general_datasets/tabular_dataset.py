from .file_dataset import FileDataset

class TabSepFileDataset(FileDataset):
    def get_example(self, i):
        line = super(TabSepFileDataset, self).get_example(i)
        parts = line.rstrip('\r\n').split('\t')
        return tuple(parts)


