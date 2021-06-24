from typing import List, Mapping, Generator, Tuple, Optional
from .ns_vocabulary import NSVocabulary
import torch

NullableTensor = Optional[torch.Tensor]

class _TranslationBase:
    def __init__(self):
        self.vocab: Optional[NSVocabulary] = None

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        raise NotImplementedError

    def index_with_vocab(self, vocab: NSVocabulary):
        self.vocab = vocab

    def to_tensor(self, example) -> Mapping[str, NullableTensor]:
        raise NotImplementedError

