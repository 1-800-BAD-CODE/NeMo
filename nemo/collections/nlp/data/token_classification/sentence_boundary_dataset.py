from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Union, Literal

import torch
import numpy as np
from omegaconf import MISSING

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.classes import Dataset
from nemo.core.neural_types import ChannelType, LabelsType, LengthsType, MaskType, NeuralType
from nemo.utils import logging


@dataclass
class SentenceBoundaryConfig:
    # Either a text file, or a list of text files
    text_files: Any = MISSING  # Any == Union[str, List[str]]
    tokenizer: TokenizerSpec = MISSING
    language: str = MISSING
    continuous_script: bool = False
    min_concat: int = 0
    max_concat: int = 3
    max_length: int = 128
    target_pad_id: int = -100
    seed: Optional[int] = 12345
    p_lowercase: float = 0.8
    # If set, read only up to this many lines per file
    max_input_lines: Optional[int] = None
    # Used for determining subword prefixes w.r.t. character alignment
    tokenizer_type: Literal["spe", "wpe"] = "spe"


class SentenceBoundaryDataset(Dataset):
    def __init__(self, config: SentenceBoundaryConfig):
        self._tokenizer: TokenizerSpec = config.tokenizer
        self._language = config.language
        self._target_pad_id = config.target_pad_id
        self._texts = self._parse_data(config.text_files, config.max_input_lines)
        self._rng: np.random.Generator = np.random.default_rng(seed=config.seed)
        self._min_concat = config.min_concat
        self._max_concat = config.max_concat
        self._max_length = config.max_length
        self._seed = config.seed
        self._is_continuous_script = config.continuous_script
        self._p_lowercase = config.p_lowercase
        self._tokenizer_type = config.tokenizer_type

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            # Use `ChannelType` to make `BertModel` happy (actually is a `TokenType`)
            "input_ids": NeuralType(("B", "T"), ChannelType()),
            "targets": NeuralType(("B", "T"), LabelsType()),
            "lengths": NeuralType(("B",), LengthsType()),
        }

    @property
    def language(self) -> str:
        return self._language

    def __len__(self):
        return len(self._texts)

    def __getitem__(self, idx):
        # Example will start with this index; concat 0 or more extra sentences based on settings
        extra_indices = self._choose_extra_indices()
        texts = [self._texts[i] for i in [idx] + extra_indices]
        # Maybe lower-case all inputs
        if self._rng.random() < self._p_lowercase:
            texts = [x.lower() for x in texts]
        # Make example based on the rules for this language
        if self._is_continuous_script:
            input_ids, targets = self._make_continuous_example(texts)
        else:
            input_ids, targets = self._make_non_continuous_example(texts)
        # Trim if needed
        if len(input_ids) > self._max_length:
            input_ids = input_ids[: self._max_length]
            targets = targets[: self._max_length]
            input_ids[-1] = self._tokenizer.eos_id
            targets[-1] = self._target_pad_id
        # # Always ignore the final target, since it's EOS
        # targets[-2] = self._target_pad_id
        return input_ids, targets

    def _make_non_continuous_example(self, texts: List[str]):
        """Generates an example for writing systems that follow rules similar to English"""
        all_ids: List[int] = [self._tokenizer.bos_id]
        all_targets: List[int] = [self._target_pad_id]
        for text in texts:
            ids = self._tokenizer.text_to_ids(text)
            targets = [0] * len(ids)
            targets[-1] = 1
            all_ids.extend(ids)
            all_targets.extend(targets)
        all_ids.append(self._tokenizer.eos_id)
        all_targets.append(self._target_pad_id)
        return torch.tensor(all_ids), torch.tensor(all_targets)

    def _make_continuous_example(self, texts: List[str]):
        """Generates an example when using a continuous-script writing system"""
        # Normalize the texts to ensure we can count char positions
        texts = [self._tokenizer.ids_to_text(self._tokenizer.text_to_ids(text)) for text in texts]
        texts = [x.replace(" ", "") for x in texts]
        concat_text = "".join(texts)
        tokens = self._tokenizer.text_to_tokens(concat_text)
        # Determine at which chars the sentence boundaries occur
        boundary_char_indices = []
        for text in texts:
            next_char_boundary = len(text) + (0 if not boundary_char_indices else boundary_char_indices[-1])
            boundary_char_indices.append(next_char_boundary)
        # Based on the char indices, determine at which token indices the sentence boundaries occur
        char_index = 0
        targets = [0] * len(tokens)
        for token_num, token in enumerate(tokens):
            num_chars_in_token = len(token)
            if self._tokenizer_type == "spe" and token.startswith("▁"):
                num_chars_in_token -= 1
            elif self._tokenizer_type == "wpe" and token.startswith("##"):
                num_chars_in_token -= 2
            char_index += num_chars_in_token
            if char_index >= boundary_char_indices[0]:
                del boundary_char_indices[0]
                targets[token_num] = 1
            if char_index == len(concat_text) or not boundary_char_indices:
                break
        input_ids = [self._tokenizer.bos_id]
        input_ids.extend(self._tokenizer.tokens_to_ids(tokens))
        input_ids.append(self._tokenizer.eos_id)
        # Add BOS/EOS padding to targets
        targets = [self._target_pad_id] + targets + [self._target_pad_id]
        return torch.tensor(input_ids), torch.tensor(targets)

    def _choose_extra_indices(self) -> List[int]:
        num_extras = self._rng.integers(low=self._min_concat, high=self._max_concat)
        extra_indices: List[int] = list(self._rng.integers(low=0, high=len(self), size=num_extras))
        return extra_indices

    def _parse_data(self, text_files: Union[str, List[str]], max_lines: Optional[int] = None) -> List[str]:
        if isinstance(text_files, str):
            text_files = [text_files]
        texts: List[str] = []
        for text_file in text_files:
            with open(text_file) as f:
                for i, line in enumerate(f):
                    if max_lines is not None and i > max_lines:
                        break
                    texts.append(line.strip())
        return texts

    def worker_init_fn(self, worker_id: int):
        # For multi-node, might want to pass global rank to `__init__` and use it here.
        # For dev datasets, use a seed. Workers will have unique RNGs, but generate similar examples every epoch.
        # For train datasets, use no seed and use random RNGs every worker every epoch.
        adjusted_seed = None if self._seed is None else (self._seed + worker_id)
        self._rng = np.random.default_rng(seed=adjusted_seed)
        # print(f" ******* Worker init function lang {self._language} worker id {worker_id} seed {adjusted_seed}")

    def collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs_ids_list: List[torch.tensor] = [x[0] for x in batch]
        targets_list: List[torch.Tensor] = [x[1] for x in batch]
        lengths: torch.Tensor = torch.tensor([x.shape[0] for x in inputs_ids_list])
        batch_size = lengths.shape[0]
        max_length = lengths.max()
        input_ids: torch.Tensor = torch.full(size=[batch_size, max_length], fill_value=self._tokenizer.pad_id)
        targets: torch.Tensor = torch.full(size=[batch_size, max_length], fill_value=self._target_pad_id)
        for i, length in enumerate(lengths):
            input_ids[i, :length] = inputs_ids_list[i]
            targets[i, :length] = targets_list[i]
        return input_ids, targets, lengths
