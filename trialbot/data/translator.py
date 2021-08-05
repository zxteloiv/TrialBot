from typing import List, Mapping, Generator, Tuple, Optional, Any
from trialbot.data.ns_vocabulary import NSVocabulary
from ._translation_base import _TranslationBase, NullableTensor
from .field import Field
import torch
from collections import defaultdict

class Translator(_TranslationBase):
    """
    A Translator has the full knowledge of the accompanied dataset.
    It has to provide APIs to convert a dataset example into a tensor based storage.
    """
    def batch_tensor(self, tensors: List[Mapping[str, NullableTensor]]) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError


class FieldAwareTranslator(Translator):
    def __init__(self, field_list: List[Field], vocab_fields: Optional[List[Field]] = None):
        """ Initialize the translator with some fields.
        field_list: default fields
        vocab_fields: only used to build the vocabulary indices, default to field_list if not applicable
        """
        super().__init__()
        self.fields = field_list
        self.vocab_fields = vocab_fields or field_list

    def index_with_vocab(self, vocab: NSVocabulary):
        super().index_with_vocab(vocab)
        for field in self.fields:
            field.index_with_vocab(vocab)

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        for field in self.vocab_fields:
            yield from field.generate_namespace_tokens(example)

    def to_tensor(self, example) -> Mapping[str, NullableTensor]:
        instance_fields = {}
        for field in self.fields:
            instance_fields.update(field.to_tensor(example))
        return instance_fields

    def batch_tensor(self, tensors: List[Mapping[str, NullableTensor]]) -> Mapping[str, torch.Tensor]:
        tensors = list(filter(lambda x: all(v is not None for v in x.values()), tensors))
        batch_dict = self.list_of_dict_to_dict_of_list(tensors)
        output = {}
        for field in self.fields:
            output.update(field.batch_tensor_by_key(batch_dict))
        return output

    @staticmethod
    def list_of_dict_to_dict_of_list(ld: List[Mapping[str, Any]]) -> Mapping[str, List[Any]]:
        list_by_keys = defaultdict(list)
        for d in ld:
            for k, v in d.items():
                list_by_keys[k].append(v)
        return list_by_keys


