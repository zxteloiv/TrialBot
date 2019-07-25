from typing import Callable, Tuple

from trialbot.data.dataset import Dataset
from trialbot.data.translator import Translator
from .hparamset import HyperParamSet

class Registry:
    """
    A static class contains references to all registered
    datasets, translators, and hparamsets by the user script.
    """
    _datasets = {}
    _translators = {}
    _hparamsets = {}

    @staticmethod
    def dataset(name=None):
        def decorator(dataset_fn: Callable[[], Tuple[Dataset, Dataset, Dataset]]):
            nonlocal name
            if name is None:
                name = dataset_fn.__name__
            Registry._datasets[name] = dataset_fn
            return dataset_fn
        return decorator

    @staticmethod
    def get_dataset(name) -> Tuple[Dataset, Dataset, Dataset]:
        return Registry._datasets[name]()

    @staticmethod
    def translator(name=None):
        def decorator(translator_fn: Callable[[], Translator]):
            nonlocal name
            if name is None:
                name = translator_fn.__name__
            Registry._translators[name] = translator_fn
            return translator_fn
        return decorator

    @staticmethod
    def get_translator(name) -> Translator:
        return Registry._translators[name]()

    @staticmethod
    def hparamset(name=None):
        def decorator(hparamset_fn: Callable[[], HyperParamSet]):
            nonlocal name
            if name is None:
                name = hparamset_fn.__name__
            Registry._hparamsets[name] = hparamset_fn
            return hparamset_fn
        return decorator

    @staticmethod
    def get_hparamset(name) -> HyperParamSet:
        return Registry._hparamsets[name]()
