from typing import Any, TypeVar
from collections.abc import Iterator
from .ns_vocabulary import NSVocabulary
from abc import ABC


T = TypeVar('T')


class _TranslationBase(ABC):
    def __init__(self):
        self.vocab: NSVocabulary | None = None

    def generate_namespace_tokens(self, example) -> Iterator[tuple[str, str]]:
        """
        An iterator yields each time a tuple of namespace and token strings
        """
        raise NotImplementedError

    def index_with_vocab(self, vocab: NSVocabulary):
        self.vocab = vocab

    def to_input(self, example) -> dict[str, T | None]:
        """
        Turn a single example into a dict, from string keys, to any value or None.
        The value must be recognizable by the batch function,
        while the None can indicate some irregular inputs that may be filtered during batch construction.
        """
        raise NotImplementedError
