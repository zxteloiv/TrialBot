from typing import Optional, List, Dict, Union, Any
import torch.nn
from .trial_bot import TrialBot
from trialbot.data.iterator import Iterator

class Updater:
    def __init__(self,
                 models: Union[List[torch.nn.Module], torch.nn.Module],
                 iterators: Union[List[Iterator], Iterator],
                 optims,
                 device: int = -1):
        self._models = models if isinstance(models, list) else [models]
        self._device = device
        self._iterators: List[Iterator] = iterators if isinstance(iterators, list) else [iterators]
        self._optims = optims if isinstance(optims, list) else [optims]

        self._epoch_ended = False

    def start_epoch(self):
        self._epoch_ended = False

    def __call__(self):
        """
        If called by __call__, Updater will returns a iterator.
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

    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'Updater':
        raise NotImplementedError

