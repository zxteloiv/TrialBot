import numpy
from ..iterator import Iterator
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

    def __init__(self, dataset_len: int, batch_size: int, repeat: bool = True, shuffle: bool = True):
        super(RandomIterator, self).__init__()
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.repeat: bool = repeat
        self.shuffle: bool = shuffle
        self._order = None if not self.shuffle else numpy.random.permutation(numpy.arange(self.dataset_len))
        self.current_position = 0
        self.logger = logging.getLogger(__name__)
        self.epoch = 0
        self.is_end_of_epoch = False

    def __next__(self):
        if self._epoch_size == 0 or self.batch_size == 0:
            self.logger.warning("Given dataset is empty. Nothing to yield but only raising StopIteration")
            raise StopIteration

        if not self.repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        N = self._epoch_size

        if self._order is None:
            batch = list(range(self.dataset_len))[i:i_end]
        else:
            batch = [index for index in self._order[i:i_end]]

        if i_end >= N:
            if self.repeat:
                rest = i_end - N
                if self._order is not None:
                    new_order = numpy.random.permutation(numpy.arange(self.dataset_len))
                    self._order = new_order

                if rest > 0:
                    if self._order is None:
                        batch.extend(list(range(rest)))
                    else:
                        batch.extend([index for index in self._order[:rest]])
                self.current_position = rest
            else:
                self.current_position = 0

            self.epoch += 1
            self.is_end_of_epoch = True
        else:
            self.is_end_of_epoch = False
            self.current_position = i_end

        return batch

    next = __next__

    def reset(self, skip=0):
        self.current_position = skip
        self.epoch = 0
        self.is_end_of_epoch = False

        if self.shuffle:
            self._order = numpy.random.permutation(numpy.arange(self.dataset_len))

    @property
    def _epoch_size(self):
        if self._order is None:
            return self.dataset_len
        else:
            return len(self._order)

