from typing import Optional, List, Dict, Union, Any
from trialbot.utils.move_to_device import move_to_device
import torch.nn

class Updater:
    def __init__(self,
                 models: Union[List[torch.nn.Module]],
                 iterators,
                 optims,
                 device: int = -1):
        self._models = models if isinstance(models, list) else [models]
        self._device = device
        self._iterators = iterators if isinstance(iterators, list) else [iterators]
        self._optims = optims if isinstance(optims, list) else [optims]

    def __call__(self):
        return self.update()

    def update(self):
        raise NotImplemented

class TestingUpdater(Updater):
    def update(self):
        model, iterator, device = self._models[0], self._iterators, self._device
        model.eval()
        batch = next(iterator)
        if 'target_tokens' in batch:
            del batch['target_tokens']
        if device >= 0:
            batch = move_to_device(batch, device)
        output = model(**batch)
        return output

class TrainingUpdater(Updater):
    def __init__(self, models, iterators, optims, device=-1, dry_run: bool = False):
        super(TrainingUpdater, self).__init__(models, iterators, optims, device)
        self._dry_run = dry_run

    def update(self):
        model, optim, iterator = self._models[0], self._optims[0], self._iterators[0]
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

