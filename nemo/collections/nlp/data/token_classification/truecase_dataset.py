import numpy as np
import re
import torch
from typing import Dict, List, Optional, Tuple, Union

from nemo.collections.common.tokenizers import TokenizerSpec, SentencePieceTokenizer
from nemo.core import Dataset, typecheck
from nemo.core.neural_types import ChannelType, LengthsType, NeuralType, LabelsType
from nemo.utils import logging


class InferenceTruecaseDataset(Dataset):
    """

    Args:
        tokenizer: A :class:`TokenizerSpec` for the model being used for inference.
        input_texts: An optional list of one or more strings to run inference on.
        input_file: An optional file to read lines from. Should be mutually exclusive with `input_texts`.
        max_length: The maximum length for inputs. Longer inputs will be split into multiple batch elements.
    """

    def __init__(
            self,
            tokenizer: TokenizerSpec,
            input_texts: Optional[List[str]] = None,
            input_file: Optional[str] = None,
            max_length: int = 512,
            ignore_idx: int = -100,
            max_subword_length: int = 16,
    ):
        super().__init__()
        if not ((input_texts is None) ^ (input_file is None)):
            raise ValueError(f"Need exactly one of `input_texts` or `input_file`")
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._ignore_idx = ignore_idx
        self._max_subword_length = max_subword_length
        self._using_sp = isinstance(self._tokenizer, SentencePieceTokenizer)

        self._data: List[str]
        if input_texts is not None:
            self._data = input_texts
        else:
            self._data = []
            with open(input_file) as f:  # noqa
                for line in f:
                    self._data.append(line.strip())
        logging.info(f"Inference dataset instantiated with {len(self._data)} lines of text.")

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(("B", "T"), ChannelType()),
            "lengths": NeuralType(("B",), LengthsType()),
        }

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        input_text = self._data[idx]
        # Don't add BOS/EOS yet because sequences get wrapped to max length in collate_fn
        input_ids = self._tokenizer.text_to_ids(input_text)
        return input_ids, input_text

    @typecheck()
    def collate_fn(self, batch):
        """
        Returns:
            A tuple adhering to this class's `input_types` (folded_input_ids, folded_batch_ids, lengths, input_strings)
                where `folded_input_ids` is the tensor of input tokens, `folded_batch_ids` map each batch element back
                to its original input number (for long sentences that were split), `lengths` is the length of each
                element in `folded_batch_ids`, and `input_strings` is the original texts from which the inputs were
                generated. `input_strings` is returns because some tokenizers are non-invertible, so this will preserve
                the original input texts.
        """
        input_ids_list: List[List[int]] = [x[0] for x in batch]
        lengths = torch.tensor([len(x) for x in input_ids_list])
        # Add 2 for BOS and EOS, clamp at max length
        lengths = lengths.add(2).clamp(self._max_length)
        bos = self._tokenizer.bos_id  # noqa
        eos = self._tokenizer.eos_id  # noqa
        input_ids = torch.full(
            size=[lengths.shape[0], lengths.max()], dtype=torch.long, fill_value=self._tokenizer.pad_id  # noqa
        )
        for batch_idx, ids in enumerate(input_ids_list):
            ids = ids[: self._max_length - 2]
            ids = [bos] + ids + [eos]
            input_ids[batch_idx, : len(ids)] = torch.tensor(ids)
        return input_ids, lengths


class TruecaseDataset(Dataset):
    """Base class for a dataset that produces examples for punctuation restoration, true casing, and sentence-boundary
    detection.

    Args:
        language: The language code for this dataset. E.g., 'en', 'es', 'zh'. Used for logging and inferring whether
            this dataset is for a continuous-script language.
        tokenizer: Text tokenizer. Can be set later, e.g., after an NLP model initializes its BertModule, but must be
            set before producing examples.
        target_pad_value: Pad targets with this value, and use it to indicate ignored tokens (e.g. uncased tokens for
            true casing). Should be the same value used in the loss function to ignore.
    """

    def __init__(
            self,
            language: str = "unk",
            tokenizer: Optional[TokenizerSpec] = None,
            target_pad_value: int = -100,
            rng_seed: Optional[int] = None,
            max_subtoken_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._language = language
        self._target_pad_value = target_pad_value
        self._rng_seed = rng_seed
        self._rng = np.random.default_rng(seed=self._rng_seed)
        self._max_token_len = max_subtoken_length
        # Call setter
        self.tokenizer = tokenizer

    @property
    def tokenizer(self) -> TokenizerSpec:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: TokenizerSpec):
        self._tokenizer = tokenizer
        if tokenizer is not None:
            self._using_sp = isinstance(self._tokenizer, SentencePieceTokenizer)
            if self._max_token_len is None:
                if not self._using_sp:
                    # Should skip special tokens, but in most cases they are shorter than longest tokens anyway
                    self._max_token_len = max(len(x) for x in self.tokenizer.vocab)  # noqa
                else:
                    # SentencePiece model - AutoTokenizer doesn't have 'vocab' attr for some SP models
                    vocab_size = tokenizer.vocab_size  # noqa
                    self._max_token_len = max(len(self.tokenizer.ids_to_tokens([idx])[0]) for idx in range(vocab_size))

    @property
    def language(self) -> str:
        return self._language

    def worker_init_fn(self, worker_id: int):
        # For dev sets, best to use a seed for consistent evaluations. For training, use None.
        seed = None if self._rng_seed is None else (self._rng_seed + worker_id)
        self._rng = np.random.default_rng(seed=seed)

    def __getitem__(self, index):
        """Implemented by derived classes """
        raise NotImplementedError()

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(("B", "T"), ChannelType()),
            "targets": NeuralType(("B", "T", "D"), LabelsType()),  # D == max_subtoken_len
            "lengths": NeuralType(("B",), LengthsType()),
        }

    def _fold_char_targets(self, tokens: List[str], char_level_targets: List[int]) -> List[List[int]]:
        all_targets: List[List[int]] = []
        # For each token, make one output list
        char_index = 0
        for token in tokens:
            token_targets: List[int] = [self._target_pad_value] * self._max_token_len
            start = 0
            if self._using_sp:
                if token.startswith("▁"):
                    start = 1
            elif token.startswith("##"):
                start = 2
            for i in range(start, len(token)):
                char_target = char_level_targets[char_index]
                token_targets[i] = char_target
                char_index += 1
            all_targets.append(token_targets)
        return all_targets

    def _char_targets_to_token_targets(self, tokens: List[str], char_targets: List[int], apply_post: bool) -> List[int]:
        all_targets: List[int] = []
        # For each token, make one output list
        char_index = 0
        for token in tokens:
            if not apply_post:
                # Keep only the target for the first character of this subword
                all_targets.append(char_targets[char_index])
            # Fast-forward to skip the rest of this tokens characters
            num_chars_in_token = len(token)
            if self._using_sp:
                if token.startswith("▁"):
                    num_chars_in_token -= 1
            elif token.startswith("##"):
                num_chars_in_token -= 2
            char_index += num_chars_in_token
            if apply_post:
                # Keep only the target for the last character of this subword
                this_target = char_targets[char_index - 1]
                all_targets.append(this_target)
        return all_targets

    @typecheck()
    def collate_fn(self, batch):
        inputs = [x[0] for x in batch]
        targets_list = [x[1] for x in batch]
        lengths = torch.tensor([x.shape[-1] for x in inputs])
        batch_size = len(inputs)  # should be all the same size

        # Create empty input ID tensors and fill non-padded regions
        input_ids = torch.full(size=(batch_size, lengths.max()), fill_value=self._tokenizer.pad_id)  # noqa
        for i in range(batch_size):
            input_ids[i, : lengths[i]] = inputs[i]

        # Create empty target tensors and fill non-padded regions
        targets = torch.full(
            size=[batch_size, lengths.max(), self._max_token_len], fill_value=self._target_pad_value
        )
        for i in range(batch_size):
            targets[i, : lengths[i], :] = targets_list[i]

        return input_ids, targets, lengths


class CapTargetsGenerator:
    """Generator of true-casing examples.

    Args:
    """

    def __init__(self, ignore_idx: int = -100) -> None:
        self._ignore_idx = ignore_idx

    def _char_is_uncased(self, char: str):  # noqa
        return char.lower() == char.upper()

    def generate_targets(self, input_text: str) -> Tuple[str, List[int]]:
        """Randomly re-cased the input text for inputs, and generates targets which matches the input.

        Args:
            input_text: A plain-text string.

        Returns:
            A tuple (new_text, targets)

        """
        # Normalize spaces to allow assumptions
        input_text = re.sub(r"\s+", " ", input_text).strip()
        out_chars: List[str] = []
        targets: List[int] = []
        for input_char in input_text:
            # No targets for space
            if input_char == " ":
                out_chars.append(" ")
                continue
            if self._char_is_uncased(input_char):
                # If uncased, input is unchanged and target is ignore_index
                targets.append(self._ignore_idx)
                out_chars.append(input_char)
            elif input_char.isupper() and (len(input_char.lower()) == 1):
                # If char is upper, maybe lower-case it and make upper target.
                # Some chars lower-case into two chars; for now, deal with it by ignoring them.
                targets.append(1)
                out_chars.append(input_char.lower())
            else:
                # Otherwise, input char is unchanged and target is the input case
                targets.append(1 if input_char.isupper() else 0)
                out_chars.append(input_char)
        return "".join(out_chars), targets


class TextTruecaseDataset(TruecaseDataset):
    """Punctuation, true-casing, and sentence-boundary detection dataset that uses text files for example generation.

    Args:
        text_files: One or more plain-text files with one sentence per line. Each line should be properly true-cased
            and punctuated.
        language: Language code for this dataset.
        tokenizer: TokenizerSpec to use to tokenize the data. Can be set later, for NLP models with forced
            initialization order.
        max_length: Maximum length of any input.
        max_lines_per_eg: Uniformly choose between 1 and this many lines to use per example.
        truncate_max_tokens: If truncating an example, truncate between 1 and this many tokens.
        target_pad_value: Padding value used in the targets. Should be the ignore_idx of your loss function.
        rng_seed: Seed for the PRNG. For training, keep at None to prevent the data loader works from using the same
            extra indices each step.
    """

    def __init__(
            self,
            text_files: Union[str, List[str]],
            language: str,
            tokenizer: Optional[TokenizerSpec] = None,
            max_length: int = 512,
            min_lines_per_eg: int = 1,
            max_lines_per_eg: int = 4,
            truncate_max_tokens: int = 5,
            truncate_percentage: float = 0.0,
            target_pad_value: int = -100,
            rng_seed: Optional[int] = None,
            max_input_lines: Optional[int] = None,
    ):
        super().__init__(
            language=language,
            tokenizer=tokenizer,
            target_pad_value=target_pad_value,
            rng_seed=rng_seed,
        )
        self._text_files = [text_files] if isinstance(text_files, str) else text_files
        self._max_length = max_length
        self._max_lines_per_eg = max_lines_per_eg
        self._min_lines_per_eg = min_lines_per_eg
        self._truncate_max_tokens = truncate_max_tokens
        self._truncate_percentage = truncate_percentage

        self._data: List[str] = self._load_data(self._text_files, max_input_lines)

        self._cap_targets_gen: CapTargetsGenerator = CapTargetsGenerator()

    def _load_data(self, text_files, max_input_lines: Optional[int] = None) -> List[str]:  # noqa
        data: List[str] = []
        for text_file in text_files:
            with open(text_file) as f:
                for line in f:
                    if max_input_lines is not None and len(data) >= max_input_lines:
                        break
                    data.append(line.strip())
        return data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        # Each sequence starts with BOS and targets ignore first index
        bos = self._tokenizer.bos_id  # noqa
        eos = self._tokenizer.eos_id  # noqa
        pad = self._target_pad_value
        pad_list = [[pad] * self._max_token_len]

        # Randomly choose how many additional lines to use
        num_extra_lines = self._rng.integers(self._min_lines_per_eg - 1, self._max_lines_per_eg)
        extra_indices = list(self._rng.integers(low=0, high=len(self), size=num_extra_lines))
        # Randomly select additional indices to use
        indices_to_use = [idx] + extra_indices
        texts: List[str] = [self._data[x] for x in indices_to_use]
        # Concatenate all the texts
        concat_unpunct_text = " ".join(texts)
        # Generate true-case targets and re-case the text
        uncased_text, target_indices = self._cap_targets_gen.generate_targets(concat_unpunct_text)
        # Generate tokens
        input_tokens = self.tokenizer.text_to_tokens(uncased_text)

        # Fold true-case targets into subword-based
        targets = self._fold_char_targets(input_tokens, target_indices)
        # Trim if too long
        input_ids = self.tokenizer.tokens_to_ids(input_tokens)
        if len(input_ids) + 2 > self._max_length:
            input_ids = input_ids[: self._max_length - 2]
            targets = targets[: self._max_length - 2]
        # # Targeting final token as sentence boundary is not useful. Ignore it.
        # Add BOS/EOS and target padding for those tokens.
        input_ids = [bos] + input_ids + [eos]
        targets = pad_list + targets + pad_list

        # Convert to Tensors.
        targets_tensor = torch.tensor(targets, dtype=torch.long)
        input_ids = torch.tensor(input_ids)
        return input_ids, targets_tensor
