#
# Some code snippets are copied from chainer, whose license is therefore adhered below.
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

import six
import numpy

class Dataset:

    """The basic implementation of the dataset abstraction.

    By default, a dataset is just a holder or connector to existing data files,
    which is required to provide two functions for the interface:

    1. the `__len__` operator which will tell the total dataset size
    2. the `__getitem__` operator which supports indexing and slicing
       over the raw examples.

    The default implementation has supported slicing over the datasets,
    which only requires subclass to implement data indexing only.
    """

    def __getitem__(self, index):
        """Returns an example or a sequence of examples.

        It implements the standard Python indexing and one-dimensional integer
        array indexing. It uses the :meth:`get_example` method by default, but
        it may be overridden by the implementation to, for example, improve the
        slicing performance.

        Args:
            index (int, slice, list or numpy.ndarray): An index of an example
                or indexes of examples.

        Returns:
            If index is int, returns an example created by `get_example`.
            If index is either slice or one-dimensional list or numpy.ndarray,
            returns a list of examples created by `get_example`.

        """
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, numpy.ndarray):
            return [self.get_example(i) for i in index]
        else:
            return self.get_example(index)

    def __len__(self):
        raise NotImplementedError

    def get_example(self, i: int):
        """Returns the i-th example.
        However, we do not require an example be of any specific type.
        As long as the interpreter of the dataset could recognize it, that's enough.
        It's better for a subclass of dataset to be annotated with a proper type,
        for the sake of IDE intelligence over the accompanied interpreter,
        but this is not a must.

        Implementations should override it. It should raise :class:`IndexError`
        if the index is invalid.

        Args:
            i (int): The index of the example.

        Returns:
            The i-th example.

        """
        raise NotImplementedError
