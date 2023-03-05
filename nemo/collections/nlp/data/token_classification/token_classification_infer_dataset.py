from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Union, Iterator

import torch
from omegaconf import MISSING
from torch.utils.data import Sampler

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.classes import Dataset
from nemo.core.neural_types import ChannelType, LengthsType, NeuralType, IntType


@dataclass
class TokenClassificationInferDatasetConfig:
    # Input data can either be a list of strings, or a file containing one text per line
    input_data: Any = MISSING  # Any == Union[str, List[str]]
    tokenizer: TokenizerSpec = MISSING
    max_length: int = 128
    overlap: int = 16


class TokenClassificationInferDataset(Dataset):
    def __init__(self, config: TokenClassificationInferDatasetConfig):
        self._tokenizer: TokenizerSpec = config.tokenizer
        self._texts: List[str] = self._parse_data(config.input_data)
        self._max_length = config.max_length

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            # Use `ChannelType` to make `BertModel` happy (actually is a `TokenType`)
            "input_ids": NeuralType(("B", "T"), ChannelType()),
            # Length of each index
            "lengths": NeuralType(("B",), LengthsType()),
            # Because we wrap long inputs, we return the original index of each input.
            # "batch_indices": NeuralType(("B",), IntType()),
        }

    def __len__(self):
        return len(self._texts)

    def __getitem__(self, idx) -> List[int]:
        text = self._texts[idx]
        tokens: List[int] = self._tokenizer.text_to_ids(text)
        input_ids = [self._tokenizer.bos_id] + tokens + [self._tokenizer.eos_id]
        return input_ids

    def _parse_data(self, input_data: Union[str, List[str]]) -> List[str]:
        texts: List[str]
        if isinstance(input_data, str):
            with open(input_data) as f:
                texts = [x.strip() for x in f.readlines()]
        else:
            texts = input_data
        return texts

    def collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs_ids_list: List[List[int]] = batch  # [x[0] for x in batch]
        # batch_indices: List[int] = [x[1] for x in batch]
        lengths: torch.Tensor = torch.tensor([len(x) for x in inputs_ids_list])
        batch_size = lengths.shape[0]
        max_length = lengths.max()
        input_ids: torch.Tensor = torch.full(size=[batch_size, max_length], fill_value=self._tokenizer.pad_id)
        for i, length in enumerate(lengths):
            input_ids[i, :length] = torch.tensor(inputs_ids_list[i])
        return input_ids, lengths
