import logging
from typing import Optional, List, Dict, Union, Any, Sequence
from trialbot.utils.move_to_device import move_to_device
import torch.nn
from ..trial_bot import TrialBot
from trialbot.data.iterators import RandomIterator
from ..updater import Updater

class TrainingUpdater(Updater):
    def __init__(self, dataset, translator, models, iterators, optims, device=-1, dry_run: bool = False, grad_clip_value: float = 0.):
        super().__init__(models, iterators, optims, device)
        self._dry_run = dry_run
        self.dataset = dataset
        self.translator = translator
        self.grad_clip_value = grad_clip_value

    def update_epoch(self):
        model, optim, iterator = self._models[0], self._optims[0], self._iterators[0]
        if iterator.is_end_of_epoch:
            self.stop_epoch()

        device = self._device
        model.train()
        optim.zero_grad()
        indices: Sequence[int] = next(iterator)
        tensor_list = [self.translator.to_tensor(self.dataset[index]) for index in indices]
        try:
            batch = self.translator.batch_tensor(tensor_list)
        except Exception as e:
            logging.getLogger(self.__class__.__name__).warning(
                f'skipping the preprocessing of the batch which raises an exception ... {str(e)}'
            )
            return None

        if device >= 0:
            batch = move_to_device(batch, device)

        output = model(**batch)
        if not self._dry_run:
            loss = output['loss']
            loss.backward()

            if self.grad_clip_value > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(), self.grad_clip_value)
            optim.step()
        return output

    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'TrainingUpdater':
        args, p, model = bot.args, bot.hparams, bot.model
        logger = bot.logger

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
        iterator = RandomIterator(len(bot.train_set), bot.hparams.batch_sz, shuffle=shuffle_iter, repeat=repeat_iter)
        if args.debug and args.skip:
            iterator.reset(args.skip)

        updater = cls(bot.train_set, bot.translator, model, iterator, optim, device, dry_run,
                      grad_clip_value=p.GRAD_CLIPPING)
        return updater


