from trialbot.utils.move_to_device import move_to_device
from ..trial_bot import TrialBot
from trialbot.data.iterators import RandomIterator
from ..updater import Updater
from trialbot.data.translator import Translator

class TestingUpdater(Updater):
    def __init__(self, dataset, translator: Translator, models, iterators, optims, device=-1):
        super().__init__(models, iterators, optims, device)
        self.dataset = dataset
        self.translator: Translator = translator

    def update_epoch(self):
        model, iterator, device = self._models[0], self._iterators[0], self._device
        model.eval()

        indices = next(iterator)
        tensor_list = [self.translator.to_tensor(self.dataset[index]) for index in indices]
        batch = self.translator.batch_tensor(tensor_list)

        if iterator.is_end_of_epoch:
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
        iterator = RandomIterator(len(self.test_set), hparams.batch_sz, shuffle=False, repeat=False)
        updater = cls(self.test_set, bot.translator, model, iterator, None, device)
        return updater

