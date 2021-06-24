from typing import List, Mapping, Optional
import torch
from ._translation_base import _TranslationBase, NullableTensor

class Field(_TranslationBase):
    """
    A field is not actually a translator, this is experimental
    and will later be refactored to inherit from another interface.
    """
    def batch_tensor_by_key(self,
                            tensors_by_keys: Mapping[str, List[NullableTensor]]
                            ) -> Mapping[str, torch.Tensor]:
        """
        A normal translator accept a batch of tensors,
        which is an iterable collection containing the instances from Translator.to_tensor interface.
        A field will handle the tensor list by some predefined key,
        but will accept all the possible tensors batch dict by the keys.
        :param tensors_by_keys:
        :return:
        """
        raise NotImplementedError

