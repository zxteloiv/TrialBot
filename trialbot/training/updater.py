
class Updater:
    """
    The most important function of the updater is designed to handle the life term of iterators.

    A TrialBot is obliged to control the component objects,
    including runtime args, hyper-parameters, models, datasets, translators, vocab, and so on.
    It is a easy and straightforward bundle to use in a trial or a few trials.

    However, the iterators are short-term compared with the others.
    They are only expected to work with the main training loops.

    To extract the complexity of iterators with other components,
    the updater (inspired by the Chainer, one of the pioneer deep learning frameworks)
    is added as an abstraction layer to manage the iterators during training loops.

    That's why the training iteration is not considered as a function
    that can be registered to the TrialBot event engine,
    but is implemented by the update_epoch function within the updater class.

    Furthermore, an iterator will define the randomness or other criterion
    to draw samples from the dataset, which however again depends on the end usage.
    Therefore it had better to initialize an iterator within the subclass,
    not in the base updater.
    """
    def __init__(self):
        self._epoch_ended = False

    def start_epoch(self):
        self._epoch_ended = False

    def __call__(self):
        """
        If called by __call__, Updater will returns an iterator.
        Therefore supporting the following paradigm for training or testing loops.

        ```python
        for output in updater():
            # do stuff with output
        ```

        The effect is the same as iter(updater).

        If you want only the next example, use next(updater) instead.

        :return:
        """
        yield from self

    def __iter__(self):
        """Return self. Support iteration for only once."""
        return self

    def __next__(self):
        if self._epoch_ended:
            raise StopIteration

        return self.update_epoch()

    def update_epoch(self):
        """
        Keep updating until the epoch ends.
        When the epoch has ended, the method should call stop_epoch.
        :return:
        """
        raise NotImplementedError

    def stop_epoch(self):
        self._epoch_ended = True
