import logging
from collections.abc import Sequence
from trialbot.utils.move_to_device import move_to_device
import torch.nn
from ..trial_bot import TrialBot
from trialbot.data.iterators import RandomIterator
from ..updater import Updater
from ..select_optims import torch_optim_cls
from ...data import Iterator, Dataset
from ...data.translator import Translator


class BatchMixin:
    iterator: Iterator
    dataset: Dataset
    translator: Translator
    device: int

    def next_batch(self) -> dict | None:
        indices: Sequence[int] = next(self.iterator)
        input_list = [self.translator.to_input(self.dataset[index]) for index in indices]
        try:
            batch = self.translator.build_batch(input_list)
        except Exception as e:
            logging.getLogger(self.__class__.__name__).warning(
                f'skipping the preprocessing of the batch which raises an exception ... {str(e)}'
            )
            return None

        if self.device >= 0:
            batch = move_to_device(batch, self.device)

        return batch


class TrainingUpdater(Updater, BatchMixin):
    def __init__(self, dataset, translator, model, iterator, optim,
                 device: int = -1, grad_clip_value: float = 0.):
        """
        Assuming the training operates on only one dataset, one model, one iterator,
        and one optimizer.
        """
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.iterator = iterator
        self.translator = translator
        self.grad_clip_value = grad_clip_value
        self.device = device
        self.optim = optim

    def update_epoch(self):
        model = self.model
        model.train()
        batch = self.next_batch()
        if batch is None:
            return None

        output = model(**batch)
        self.complete_iteration(output.get('loss'))

        if self.iterator.is_end_of_epoch:
            self.stop_epoch()

        return output

    def complete_iteration(self, loss):
        if loss is None:
            return

        optim = self.optim
        optim.zero_grad()
        loss.backward()
        if self.grad_clip_value > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip_value)
        optim.step()

    @classmethod
    def from_bot(cls, bot: TrialBot, optim_cls=None) -> 'TrainingUpdater':
        """
        optim_cls: the given class must be pre-filled with kwargs
        """
        args, p, model = bot.args, bot.hparams, bot.model
        logger = bot.logger

        optim_cls = optim_cls if optim_cls is not None else torch_optim_cls(p)
        optim = optim_cls(model.parameters())
        logger.info(f'Using the optimizer {optim.__class__.__name__}: {str(optim)}')

        repeat_iter = not args.debug
        shuffle_iter = not args.debug

        iterator = RandomIterator(len(bot.train_set), bot.hparams.batch_sz,
                                  shuffle=shuffle_iter, repeat=repeat_iter)
        if args.debug and args.skip:
            iterator.reset(args.skip)

        updater = cls(bot.train_set, bot.translator, model, iterator, optim,
                      device=args.device, grad_clip_value=p.GRAD_CLIPPING)
        return updater
