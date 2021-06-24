#
# This interface code is originated from chainer, whose license is therefore adhered below.
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
from typing import Sequence

class Iterator(object):
    """Base class of all dataset iterators.

    Iterator iterates over the dataset, yielding a sequence of indices at each iteration.
    The indices sequence usually indicates a list of examples of a minibatch.
    Each implementation should implement an iterator protocol (e.g., the :meth:`__next__` method).

    Each implementation should provide the following attributes (not needed to be writable).

    - ``epoch``: Number of completed sweeps over the dataset.
    - ``is_end_of_epoch``: ``True`` if the batch indices returned is the end of the epoch.

    """
    def __init__(self):
        self.is_end_of_epoch = False
        self.epoch = 0

    def __iter__(self):
        """Returns self."""
        return self

    def __next__(self) -> Sequence[int]:
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


