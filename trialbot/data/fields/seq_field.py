from typing import Any
from ..field import Field, T
from collections.abc import Iterator
import torch
from trialbot.data import START_SYMBOL, END_SYMBOL
from itertools import product
from torch.nn.utils.rnn import pad_sequence


class SeqField(Field):
    def get_sent(self, example):
        return example.get(self.source_key)

    def generate_namespace_tokens(self, example: Any) -> Iterator[tuple[str, str]]:
        if self.add_start_end_toks:
            yield from product([self.ns], [START_SYMBOL, END_SYMBOL])

        seq_raw = self.get_sent(example)
        if seq_raw is not None:
            yield from product([self.ns], (x.lower() if self.lower_case else x
                                           for x in self.split(seq_raw)))

    def to_input(self, example) -> dict[str, T | None]:
        seq_raw = self.get_sent(example)
        if seq_raw is None:
            return {self.renamed_key: torch.tensor([self.padding])}

        seq_toks = self.split(seq_raw)
        if len(seq_toks) == 0:
            return {self.renamed_key: torch.tensor([self.padding])}

        if self.max_seq_len > 0:
            seq_toks = seq_toks[:self.max_seq_len]

        if self.lower_case:
            seq_toks = [x.lower() if self.lower_case else x for x in seq_toks]

        if self.add_start_end_toks:
            seq_toks = [START_SYMBOL] + seq_toks + [END_SYMBOL]

        seq_toks = [self.vocab.get_token_index(tok, self.ns) for tok in seq_toks]
        seq_tensor = torch.tensor(seq_toks)
        return {self.renamed_key: seq_tensor}

    def build_batch_by_key(self, input_dict: dict[str, list[T]]) -> dict[str, torch.Tensor | list[T]]:
        tensor_list = input_dict.get(self.renamed_key)
        if tensor_list is None or len(tensor_list) == 0:
            raise KeyError(f'Empty field key {self.renamed_key} confronted. Failed to build the instance tensors')

        batch_tensor = pad_sequence(tensor_list, batch_first=True, padding_value=self.padding)
        return {self.renamed_key: batch_tensor}

    def __init__(self, source_key: str,
                 renamed_key: str = None,
                 namespace: str = None,
                 split_fn=None,
                 add_start_end_toks: bool = True,
                 padding_id: int = 0,
                 max_seq_len: int = 0,
                 use_lower_case: bool = True,
                 ):
        super().__init__()
        self.split = split_fn or (lambda s: s if isinstance(s, list) else s.split())
        self.source_key = source_key
        self.ns = namespace or source_key
        self.renamed_key = renamed_key or source_key
        self.add_start_end_toks = add_start_end_toks
        self.padding = padding_id
        self.max_seq_len = max_seq_len
        self.lower_case = use_lower_case
