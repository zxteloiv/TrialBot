from typing import List, Mapping, Generator, Tuple
from trialbot.data.ns_vocabulary import NSVocabulary
import torch

class Translator:
    """
    A Translator has the full knowledge of the accompanied dataset.
    It has to provide APIs to convert a dataset example into a tensor based storage.
    """
    def __init__(self):
        self.vocab: NSVocabulary = None

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        raise NotImplemented

    def index_with_vocab(self, vocab: NSVocabulary):
        self.vocab = vocab

    def to_tensor(self, example) -> Mapping[str, torch.Tensor]:
        raise NotImplemented

    def batch_tensor(self, tensors: List[Mapping[str, torch.Tensor]]) -> Mapping[str, torch.Tensor]:
        raise NotImplemented

