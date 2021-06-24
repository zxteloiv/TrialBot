from typing import List, Mapping, Union, Callable, Any

import random

import logging
from collections import defaultdict

from .bucket_iterator import BucketIterator

class ClusterIterator(BucketIterator):
    def __init__(self,
                 dataset,
                 batch_size,
                 cluster_key: Union[str, Callable[[Any], str]],
                 repeat=True,
                 shuffle=True,
                 ):
        super().__init__(dataset, batch_size, cluster_key, repeat=repeat, shuffle=shuffle)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._clusters = None
        if isinstance(cluster_key, str):
            self._key_func = lambda x: x[cluster_key]

    def _get_batches(self):
        if self._clusters is None:
            self._clusters = defaultdict(list)
            for i, x in enumerate(self.dataset):
                key = self._key_func(x)
                self._clusters[key].append(i)

        groups = list(self._clusters.values())
        if self._shuffle:
            random.shuffle(groups)

        batches = []
        for g in groups:
            items = [i for i in g]
            random.shuffle(items)
            for i in range(0, len(items), self.batch_size):
                batch = items[i:i+self.batch_size]
                batches.append(batch)
        return batches
