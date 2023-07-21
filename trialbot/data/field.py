from abc import ABC

import torch
from ._translation_base import _TranslationBase, T


class Field(_TranslationBase, ABC):
    """
    A field is not actually a translator, this is experimental
    and will later be refactored to inherit from another interface.
    """
    def build_batch_by_key(self, input_dict: dict[str, list[T]]) -> dict[str, torch.Tensor | list[T]]:
        """
        A normal translator accept a batch of tensors,
        which is an iterable collection containing the instances from Translator.to_tensor interface.
        A field will handle the tensor list by some predefined key,
        but will accept all the possible tensors batch dict by the keys.
        :param input_dict:
        :return: a dict mapping string keys to values that can be either
                - a tensor representing a feature of the batch,
                - a list of something that ignored by the translator but usable to the model.
        """
        raise NotImplementedError
