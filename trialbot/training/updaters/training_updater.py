import logging
from collections.abc import Sequence
from trialbot.utils.move_to_device import move_to_device
from trialbot.utils.multi_gpu import (
    move_to_device_multigpu,
    parse_gpu_ids,
    setup_deepspeed,
    is_distributed,
    get_rank
)
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
    gpu_ids: list = None
    args = None

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

        # Use multi-GPU aware device movement
        batch = move_to_device_multigpu(batch, self.device, self.gpu_ids)

        return batch


class TrainingUpdater(Updater, BatchMixin):
    def __init__(self, dataset, translator, model, iterator, optim,
                 device: int = -1, grad_clip_value: float = 0.,
                 args=None, gpu_ids=None):
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
        self.args = args
        self.gpu_ids = gpu_ids
        self.deepspeed_engine = None
        
        # Initialize DeepSpeed if needed
        if args and args.deepspeed:
            self._init_deepspeed()

    def _init_deepspeed(self):
        """Initialize DeepSpeed engine."""
        try:
            import deepspeed
            from trialbot.utils.multi_gpu import setup_deepspeed
            
            # Setup DeepSpeed
            self.model, self.optim, _ = setup_deepspeed(
                self.args, self.model, self.optim
            )
            self.deepspeed_engine = self.model
            logging.getLogger(self.__class__.__name__).info(
                "DeepSpeed engine initialized"
            )
        except ImportError:
            logging.getLogger(self.__class__.__name__).warning(
                "DeepSpeed not installed. Continuing without DeepSpeed."
            )

    def update_epoch(self):
        model = self.model
        model.train()
        batch = self.next_batch()
        if batch is None:
            return None

        # For DeepSpeed, use engine forward
        if self.deepspeed_engine is not None:
            output = self.deepspeed_engine(**batch)
        else:
            output = model(**batch)
        
        self.complete_iteration(output.get('loss'))

        if self.iterator.is_end_of_epoch:
            self.stop_epoch()

        return output

    def complete_iteration(self, loss):
        if loss is None:
            return

        # Handle DeepSpeed backward
        if self.deepspeed_engine is not None:
            self.deepspeed_engine.backward(loss)
            self.deepspeed_engine.step()
            return

        # Standard PyTorch backward
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

        # Adjust batch size for multi-GPU
        batch_size = p.batch_sz
        gpu_ids = parse_gpu_ids(args.gpus)
        
        # For distributed training, adjust batch size per GPU
        if is_distributed():
            try:
                import torch.distributed as dist
                world_size = dist.get_world_size()
                # Divide batch size by world size for data parallelism
                if world_size > 1:
                    batch_size = max(1, batch_size // world_size)
                    logger.info(f"Adjusted batch size to {batch_size} per GPU (world_size={world_size})")
            except:
                # Fallback to using gpu_ids length
                if gpu_ids and len(gpu_ids) > 1:
                    batch_size = max(1, batch_size // len(gpu_ids))
                    logger.info(f"Adjusted batch size to {batch_size} per GPU (using {len(gpu_ids)} GPUs)")

        iterator = RandomIterator(len(bot.train_set), batch_size,
                                  shuffle=shuffle_iter, repeat=repeat_iter)
        if args.debug and args.skip:
            iterator.reset(args.skip)

        updater = cls(bot.train_set, bot.translator, model, iterator, optim,
                      device=args.device, grad_clip_value=p.GRAD_CLIPPING,
                      args=args, gpu_ids=gpu_ids)
        return updater
