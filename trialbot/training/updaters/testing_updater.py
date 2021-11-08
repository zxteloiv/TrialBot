from trialbot.utils.move_to_device import move_to_device
import logging
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
        try:
            batch = self.translator.batch_tensor(tensor_list)
        except Exception as e:
            logging.getLogger(self.__class__.__name__).warning(
                f'skipping the preprocessing of the batch which raises an exception ... {str(e)}'
            )
            return None

        if iterator.is_end_of_epoch:
            self.stop_epoch()
        if device >= 0:
            batch = move_to_device(batch, device)
        output = model(**batch)
        return output

    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'TestingUpdater':
        args, model, hparams = bot.args, bot.model, bot.hparams
        device = args.device
        dataset = bot.dev_set if args.dev else bot.test_set
        iterator = RandomIterator(len(dataset), hparams.batch_sz, shuffle=False, repeat=False)
        updater = cls(dataset, bot.translator, model, iterator, None, device)
        return updater

