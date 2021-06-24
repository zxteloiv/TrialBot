from typing import List, Mapping, Union, Callable, Any, Sequence, Optional

import random

from trialbot.data.iterator import Iterator

import logging

class BucketIterator(Iterator):
    def __init__(self,
                 dataset,
                 batch_size,
                 key: Union[str, Callable[[Any], float]], # only one key used to rank
                 sort_noise: float = .1,
                 repeat=True,
                 drop_last: bool = False,
                 shuffle=True,
                 ):
        super().__init__()
        self.dataset = dataset
        self._dataset_lengths = None
        self.batch_size = batch_size

        self._key_func: Callable[[...], float] = key
        if isinstance(key, str):
            self._key_func = lambda x: len(x[key])

        self.sort_noise = sort_noise

        self.drop_last = drop_last

        self.repeat = repeat
        self._shuffle = shuffle
        self.logger = logging.getLogger(self.__class__.__name__)
        self._current_bucket = 0
        self.epoch = 0
        self._batches: Optional[List[List[int]]] = None

    def __next__(self) -> Sequence[int]:
        if self.epoch_size == 0 or self.batch_size <= 0:
            self.logger.warning("Given dataset is empty. Nothing to yield but only raising StopIteration")
            raise StopIteration

        if not self.repeat and self.epoch > 0:
            raise StopIteration

        if self._batches is None:
            self._batches = self._get_batches()

        indices = self._batches[self._current_bucket]

        self.is_end_of_epoch = False
        self._current_bucket += 1

        if self._current_bucket >= len(self._batches):
            self._current_bucket = 0
            self.epoch += 1
            # the current batch is the last of the epoch, and the is_end_of_epoch flag is set True
            # so the epoch training must not stop immediately but after processing the returned batch.
            self.is_end_of_epoch = True
            self._batches = None    # buckets will be reshuffled for the next turn

        return indices

    @staticmethod
    def add_uniform_noise_to_value(value: int, noise_param: float):
        if noise_param > 0:
            noise_value = value * noise_param
            noise = random.uniform(-noise_value, noise_value)
            return value + noise
        else:
            return value

    def _argsort_dataset(self) -> List[int]:
        if self._dataset_lengths is None:
            self._dataset_lengths = [self._key_func(x) for x in self.dataset]

        noisy_lengths = [self.add_uniform_noise_to_value(v, self.sort_noise) for v in self._dataset_lengths]
        indices = sorted(range(self.epoch_size), key=lambda i: noisy_lengths[i])
        return indices

    def _get_batches(self) -> List[List[int]]:
        indices = self._argsort_dataset()
        buckets = []
        # split into buckets and randomly shuffle the buckets (the groups)
        for i in range(0, len(indices), self.batch_size):
            group = indices[i:i+self.batch_size]
            if len(group) < self.batch_size and self.drop_last:
                continue
            buckets.append(group)

        if self._shuffle:
            random.shuffle(buckets)

        return buckets

    next = __next__

    @property
    def epoch_size(self):
        return len(self.dataset)

