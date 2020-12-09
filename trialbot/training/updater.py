from typing import Optional, List, Dict, Union, Any
from trialbot.utils.move_to_device import move_to_device
import torch.nn
from .trial_bot import TrialBot
from trialbot.data import Iterator, RandomIterator

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

class TestingUpdater(Updater):
    def update_epoch(self):
        model, iterator, device = self._models[0], self._iterators[0], self._device
        model.eval()
        batch = next(iterator)
        if iterator.is_new_epoch:
            self.stop_epoch()
        if device >= 0:
            batch = move_to_device(batch, device)
        output = model(**batch)
        return output

    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'TestingUpdater':
        self = bot
        args, model = self.args, self.model
        device = args.device

        hparams, model = self.hparams, self.model
        iterator = RandomIterator(self.test_set, hparams.batch_sz, self.translator, shuffle=False, repeat=False)
        updater = cls(model, iterator, None, device)
        return updater

class TrainingUpdater(Updater):
    def __init__(self, models, iterators, optims, device=-1, dry_run: bool = False):
        super(TrainingUpdater, self).__init__(models, iterators, optims, device)
        self._dry_run = dry_run

    def update_epoch(self):
        model, optim, iterator = self._models[0], self._optims[0], self._iterators[0]
        if iterator.is_new_epoch:
            self.stop_epoch()

        device = self._device
        model.train()
        optim.zero_grad()
        batch: Dict[str, torch.Tensor] = next(iterator)

        if device >= 0:
            batch = move_to_device(batch, device)

        output = model(**batch)
        if not self._dry_run:
            loss = output['loss']
            loss.backward()
            optim.step()
        return output

    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'TrainingUpdater':
        self = bot
        args, p, model = self.args, self.hparams, self.model
        logger = self.logger

        if hasattr(p, "OPTIM") and isinstance(p.OPTIM, str) and p.OPTIM.lower() == "sgd":
            logger.info(f"Using SGD optimzer with lr={p.SGD_LR}")
            optim = torch.optim.SGD(model.parameters(), p.SGD_LR, weight_decay=p.WEIGHT_DECAY)
        elif hasattr(p, "OPTIM") and isinstance(p.OPTIM, str) and p.OPTIM.lower() == "radam":
            from radam import RAdam
            optim = RAdam(model.parameters(), lr=p.ADAM_LR, weight_decay=p.WEIGHT_DECAY)
            logger.info(f"Using RAdam optimzer with lr={p.ADAM_LR}")
        else:
            logger.info(f"Using Adam optimzer with lr={p.ADAM_LR} and beta={str(p.ADAM_BETAS)}")
            optim = torch.optim.Adam(model.parameters(), p.ADAM_LR, p.ADAM_BETAS, weight_decay=p.WEIGHT_DECAY)

        device = args.device
        dry_run = args.dry_run
        repeat_iter = not args.debug
        shuffle_iter = not args.debug
        iterator = RandomIterator(self.train_set, self.hparams.batch_sz, self.translator,
                                  shuffle=shuffle_iter, repeat=repeat_iter)
        if args.debug and args.skip:
            iterator.reset(args.skip)

        updater = cls(model, iterator, optim, device, dry_run)
        return updater


