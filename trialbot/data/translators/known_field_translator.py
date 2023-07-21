from ..translator import FieldAwareTranslator, T
import torch


class _DictKeyRefWrapper(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_set = set()

    def get(self, k):
        if k not in self:
            return None
        return self[k]

    def __getitem__(self, item):
        self.ref_set.add(item)
        return super().__getitem__(item)


class KnownFieldTranslator(FieldAwareTranslator):
    """
    A Translator aware of fields,
    not only translates known keys of an instance into the tensors as the specified fields,
    but collects other unknown field of an instance and arrange them into an array.
    """
    def __init__(self, accessed_field_set=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._accessed_field_set: set | None = accessed_field_set

    def to_input(self, example) -> dict[str, T | None]:
        output = {}
        if self._accessed_field_set is None:
            example = _DictKeyRefWrapper(example)
            output.update(super().to_input(example))
            self._accessed_field_set = set(example.ref_set)
        else:
            output.update(super().to_input(example))

        for field in filter(lambda k: k not in self._accessed_field_set, example.keys()):
            output[field] = example[field]

        return output

    def build_batch(self, input_list: list[dict[str, T | None]]) -> dict[str, torch.Tensor | list[T]]:
        processed_fields = self._accessed_field_set or set()
        tensors = list(filter(lambda x: all(v is not None for v in x.values()), input_list))
        batch_dict = self.list_of_dict_to_dict_of_list(tensors)
        output = {}
        for field in self.fields:
            output.update(field.build_batch_by_key(batch_dict))
        for k, v in batch_dict.items():
            if k not in processed_fields and k not in output.keys():
                output[k] = v
        return output
