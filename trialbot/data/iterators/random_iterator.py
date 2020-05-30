from __future__ import division
from typing import List, Mapping

import numpy
import torch

from ..iterator import Iterator
from ..translator import Translator

import logging


class RandomIterator(Iterator):

    """Dataset iterator that serially reads the examples.

    This is a simple implementation of :class:`~chainer.dataset.Iterator`
    that just visits each example in either the order of indexes or a shuffled
    order.

    To avoid unintentional performance degradation, the ``shuffle`` option is
    set to ``True`` by default. For validation, it is better to set it to
    ``False`` when the underlying dataset supports fast slicing. If the
    order of examples has an important meaning and the updater depends on the
    original order, this option should be set to ``False``.

    Args:
        dataset: Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch. Otherwise, examples are extracted in the
            order of indexes. If ``None`` and no ``order_sampler`` is given,
            the behavior is the same as the case with ``shuffle=True``.
    """

    def __init__(self, dataset, batch_size, translator,
                 repeat=True, shuffle=True):
        super(RandomIterator, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.translator: Translator = translator
        self._repeat = repeat
        self._shuffle = shuffle
        self._order = None
        self.logger = logging.getLogger(__name__)

        self.reset()

    def __next__(self):
        if self._epoch_size == 0:
            self.logger.warning("Given dataset is empty. Nothing to yield but only raising StopIteration")
            raise StopIteration

        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        i = self.current_position
        i_end = i + self.batch_size
        N = self._epoch_size

        if self._order is None:
            batch = self.dataset[i:i_end]
        else:
            batch = [self.dataset[index] for index in self._order[i:i_end]]

        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                if self._order is not None:
                    new_order = numpy.random.permutation(numpy.arange(len(self.dataset)))
                    self._order = new_order

                if rest > 0:
                    if self._order is None:
                        batch.extend(self.dataset[:rest])
                    else:
                        batch.extend([self.dataset[index]
                                      for index in self._order[:rest]])
                self.current_position = rest
            else:
                self.current_position = 0

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return self.batch_to_tensors(batch)

    next = __next__

    def batch_to_tensors(self, batch: List) -> Mapping[str, torch.Tensor]:
        tensor_list = [self.translator.to_tensor(example) for example in batch]
        tensor = self.translator.batch_tensor(tensor_list)
        return tensor

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / self._epoch_size

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def reset(self, skip=0):
        self.current_position = skip
        self.epoch = 0
        self.is_new_epoch = False

        # use -1 instead of None internally.
        self._previous_epoch_detail = -1.

        if self._shuffle:
            self._order = numpy.random.permutation(numpy.arange(len(self.dataset)))

    @property
    def _epoch_size(self):
        if self._order is None:
            return len(self.dataset)
        else:
            return len(self._order)

    @property
    def repeat(self):
        return self._repeat

