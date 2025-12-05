from ..trial_bot import TrialBot
from trialbot.data.iterators import RandomIterator
from ..updater import Updater
from trialbot.data.translator import Translator
from trialbot.utils.multi_gpu import (
    move_to_device_multigpu,
    parse_gpu_ids,
    is_distributed,
    get_world_size
)
from .training_updater import BatchMixin


class TestingUpdater(Updater, BatchMixin):
    def __init__(self, dataset, translator, model, iterator, device: int = -1,
                 args=None, gpu_ids=None):
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.iterator = iterator
        self.device = device
        self.translator: Translator = translator
        self.args = args
        self.gpu_ids = gpu_ids

    def next_batch(self):
        # Override to use multi-GPU aware device movement
        from collections.abc import Sequence
        import logging
        
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

    def update_epoch(self):
        model = self.model
        model.eval()

        batch = self.next_batch()
        if batch is None:
            return None

        output = model(**batch)

        if self.iterator.is_end_of_epoch:
            self.stop_epoch()
        return output

    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'TestingUpdater':
        args, model, hparams = bot.args, bot.model, bot.hparams
        dataset = bot.dev_set if args.dev else bot.test_set
        gpu_ids = parse_gpu_ids(args.gpus)
        
        # Adjust batch size for multi-GPU
        batch_size = hparams.batch_sz
        
        # For distributed training, adjust batch size per GPU
        if is_distributed():
            try:
                import torch.distributed as dist
                world_size = dist.get_world_size()
                # Divide batch size by world size for data parallelism
                if world_size > 1:
                    batch_size = max(1, batch_size // world_size)
                    bot.logger.info(f"Adjusted test batch size to {batch_size} per GPU (world_size={world_size})")
            except:
                pass
        
        iterator = RandomIterator(len(dataset), batch_size, shuffle=False, repeat=False)
        updater = cls(dataset, bot.translator, model, iterator, args.device,
                      args=args, gpu_ids=gpu_ids)
        return updater
