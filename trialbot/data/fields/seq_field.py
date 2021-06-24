from typing import List, Mapping, Generator, Tuple, Optional, Any, Literal
from ..field import Field, NullableTensor
import torch
from trialbot.data import START_SYMBOL, END_SYMBOL, PADDING_TOKEN, NSVocabulary
from itertools import product
from torch.nn.utils.rnn import pad_sequence

class SeqField(Field):
    def get_sent(self, example):
        return example.get(self.source_key)

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        if self.add_start_end_toks:
            yield from product([self.ns], [START_SYMBOL, END_SYMBOL])

        seq_raw = self.get_sent(example)
        if seq_raw is not None:
            yield from product([self.ns], (x.lower() if self.lower_case else x for x in self.split(seq_raw)))

    def to_tensor(self, example) -> Mapping[str, NullableTensor]:
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

    def batch_tensor_by_key(self,
                            tensors_by_keys: Mapping[str, List[NullableTensor]]
                            ) -> Mapping[str, torch.Tensor]:
        tensor_list = tensors_by_keys.get(self.renamed_key)
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
