from typing import Any
from trialbot.data.ns_vocabulary import NSVocabulary
from ._translation_base import _TranslationBase, T, ABC
from .field import Field
import torch
from collections import defaultdict
from collections.abc import Iterator


class Translator(_TranslationBase, ABC):
    """
    A Translator has the full knowledge of the accompanied dataset.
    It has to provide APIs to convert a dataset example into a tensor based storage.
    """
    def build_batch(self, input_list: list[dict[str, T]]) -> dict[str, torch.Tensor | list[T]]:
        raise NotImplementedError


class FieldAwareTranslator(Translator):
    def __init__(self,
                 field_list: list[Field],
                 vocab_fields: list[Field] | None = None,
                 filter_none: bool = True):
        """ Initialize the translator with some fields.
        field_list: default fields
        vocab_fields: only used to build the vocabulary indices, default to field_list if not applicable
        filter_none: remove the example with a None value for any field, default is True.
        """
        super().__init__()
        self.fields = field_list
        self.vocab_fields = field_list if vocab_fields is None else vocab_fields
        self.filter_none = filter_none

    def index_with_vocab(self, vocab: NSVocabulary):
        super().index_with_vocab(vocab)
        for field in self.fields:
            field.index_with_vocab(vocab)

    def generate_namespace_tokens(self, example) -> Iterator[tuple[str, str]]:
        for field in self.vocab_fields:
            yield from field.generate_namespace_tokens(example)

    def to_input(self, example) -> dict[str, T | None]:
        instance_fields = {}
        for field in self.fields:
            instance_fields.update(field.to_input(example))
        return instance_fields

    def build_batch(self, input_list: list[dict[str, T | None]]) -> dict[str, torch.Tensor | list[T]]:
        if self.filter_none:
            input_list = [x for x in input_list if all(v is not None for v in x.values())]
        batch_dict = self.list_of_dict_to_dict_of_list(input_list)
        output = {}
        for field in self.fields:
            output.update(field.build_batch_by_key(batch_dict))
        return output

    @staticmethod
    def list_of_dict_to_dict_of_list(ld: list[dict]) -> dict[Any, list]:
        dl = defaultdict(list)
        for d in ld:
            for k, v in d.items():
                dl[k].append(v)
        return dl


