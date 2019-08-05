#
# This interface code is copied from chainer, whose license is therefore adhered below.
#
# Copyright (c) 2015 Preferred Infrastructure, Inc.
# Copyright (c) 2015 Preferred Networks, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
from typing import Mapping
import torch

class Iterator(object):

    """Base class of all dataset iterators.

    Iterator iterates over the dataset, yielding a minibatch at each
    iteration. Minibatch is a list of examples. Each implementation should
    implement an iterator protocol (e.g., the :meth:`__next__` method).

    Note that, even if the iterator supports setting the batch size, it does
    not guarantee that each batch always contains the same number of examples.
    For example, if you let the iterator to stop at the end of the sweep, the
    last batch may contain a fewer number of examples.

    The interface between the iterator and the underlying dataset is not fixed,
    and up to the implementation.

    Each implementation should provide the following attributes (not needed to
    be writable).

    - ``batch_size``: Number of examples within each minibatch.
    - ``epoch``: Number of completed sweeps over the dataset.
    - ``epoch_detail``: Floating point number version of the epoch. For
      example, if the iterator is at the middle of the dataset at the third
      epoch, then this value is 2.5.
    - ``previous_epoch_detail``: The value of ``epoch_detail`` at the previous
      iteration. This value is ``None`` before the first iteration.
    - ``is_new_epoch``: ``True`` if the epoch count was incremented at the last
      update.

    Each implementation should also support serialization to resume/suspend the
    iteration.

    """
    def __init__(self):
        self.is_new_epoch = False

    def __iter__(self):
        """Returns self."""
        return self

    def __next__(self) -> Mapping[str, torch.Tensor]:
        """Returns the next batch.

        This is a part of the iterator protocol of Python. It may raise the
        :class:`StopIteration` exception when it stops the iteration.

        """
        raise NotImplementedError

    def next(self):
        """Python2 alternative of ``__next__``.

        It calls :meth:`__next__` by default.
        """
        return self.__next__()

    def finalize(self):
        """Finalizes the iterator and possibly releases the resources.

        This method does nothing by default. Implementation may override it to
        better handle the internal resources.

        """
        pass

    def __enter__(self):
        """With statement context manager method

        This method does nothing by default. Implementation may override it to
        better handle the internal resources by with statement.

        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """With statement context manager method

        This method does nothing by default. Implementation may override it to
        better handle the internal resources by with statement.

        """
        return None


