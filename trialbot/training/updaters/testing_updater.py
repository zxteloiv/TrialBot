from ..trial_bot import TrialBot
from trialbot.data.iterators import RandomIterator
from ..updater import Updater
from trialbot.data.translator import Translator
from .training_updater import BatchMixin


class TestingUpdater(Updater, BatchMixin):
    def __init__(self, dataset, translator, model, iterator, device: int = -1):
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.iterator = iterator
        self.device = device
        self.translator: Translator = translator

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
        iterator = RandomIterator(len(dataset), hparams.batch_sz, shuffle=False, repeat=False)
        updater = cls(dataset, bot.translator, model, iterator, args.device)
        return updater
