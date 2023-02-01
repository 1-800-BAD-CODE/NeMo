import abc
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from nemo.collections.common.tokenizers import TokenizerSpec, SentencePieceTokenizer
from nemo.core import Dataset, typecheck
from nemo.core.neural_types import ChannelType, IntType, LengthsType, NeuralType, LabelsType, MaskType
from nemo.utils import logging


class InferencePunctCapSegDataset(Dataset):
    """

    Args:
        tokenizer: A :class:`TokenizerSpec` for the model being used for inference.
        input_texts: An optional list of one or more strings to run inference on.
        input_file: An optional file to read lines from. Should be mutually exclusive with `input_texts`.
        max_length: The maximum length for inputs. Longer inputs will be split into multiple batch elements.
        fold_overlap: When folding long sequences, repeat this many tokens from the end of the previous split into the
            beginning of the next split.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        input_texts: Optional[List[str]] = None,
        input_file: Optional[str] = None,
        max_length: int = 512,
        fold_overlap: int = 16,
        ignore_idx: int = -100,
        max_subword_length: int = 16,
    ):
        super().__init__()
        if not ((input_texts is None) ^ (input_file is None)):
            raise ValueError(f"Need exactly one of `input_texts` or `input_file`")
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._fold_overlap = fold_overlap
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
            "folded_input_ids": NeuralType(("B", "T"), ChannelType()),
            "folded_batch_ids": NeuralType(("B",), IntType()),
            "lengths": NeuralType(("B",), LengthsType()),
            "punc_mask": NeuralType(("B", "T", "D"), MaskType()),  # D == max_subtoken_len
        }

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        input_text = self._data[idx]
        # Don't add BOS/EOS yet because sequences get wrapped to max length in collate_fn
        input_ids = self._tokenizer.text_to_ids(input_text)
        return input_ids, input_text

    def _fold_batch(self, input_ids: List[List[int]]) -> Tuple[torch.Tensor, ...]:
        """Folds inputs to adhere to max length"""
        out_batch_ids: List[int] = []
        out_input_ids: List[List[int]] = []
        out_lengths: List[int] = []
        bos = self._tokenizer.bos_id  # noqa
        eos = self._tokenizer.eos_id  # noqa
        for batch_idx, next_input_ids in enumerate(input_ids):
            start = 0
            while True:
                stop = min(start + self._max_length - 2, len(next_input_ids))
                subsegment_ids = [bos] + next_input_ids[start:stop] + [eos]
                out_input_ids.append(subsegment_ids)
                out_lengths.append(len(subsegment_ids))
                out_batch_ids.append(batch_idx)
                if stop >= len(next_input_ids):
                    break
                start = stop - self._fold_overlap

        batch_ids = torch.tensor(out_batch_ids)
        lengths = torch.tensor(out_lengths)
        ids_tensor = torch.full(
            size=[lengths.shape[0], lengths.max()], dtype=torch.long, fill_value=self._tokenizer.pad_id  # noqa
        )
        punc_mask = torch.zeros(size=[lengths.shape[0], lengths.max(), self._max_subword_length])
        for batch_idx, ids in enumerate(out_input_ids):
            ids_tensor[batch_idx, : len(ids)] = torch.tensor(ids)
            tokens = self._tokenizer.ids_to_tokens(ids[1:-1])
            for token_idx, token in enumerate(tokens):
                start = 0
                if self._using_sp:
                    if token.startswith("▁"):
                        start = 1
                elif token.startswith("##"):
                    start = 2
                # Offset by one because sequences have BOS
                punc_mask[batch_idx, token_idx + 1, start : len(token) + 1] = 1

        return ids_tensor, lengths, batch_ids, punc_mask

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
        all_ids: List[List[int]] = [x[0] for x in batch]
        input_ids, lengths, batch_ids, punc_mask = self._fold_batch(all_ids)
        return input_ids, batch_ids, lengths, punc_mask


class PunctCapSegDataset(Dataset):
    """Base class for a dataset that produces examples for punctuation restoration, true casing, and sentence-boundary
    detection.

    Args:
        language: The language code for this dataset. E.g., 'en', 'es', 'zh'. Used for logging and inferring whether
            this dataset is for a continuous-script language.
        is_continuous: Whether this language is continuous. Determines whether spaces are inserted between concatenated
            sentences, etc. If not set, the language code will be compared against a list of known continuous-script
            language codes and this value will be inferred.
        tokenizer: Text tokenizer. Can be set later, e.g., after an NLP model initializes its BertModule, but must be
            set before producing examples.
        target_pad_value: Pad targets with this value, and use it to indicate ignored tokens (e.g. uncased tokens for
            true casing). Should be the same value used in the loss function to ignore.
    """

    def __init__(
        self,
        language: str = "unk",
        is_continuous: bool = None,
        tokenizer: Optional[TokenizerSpec] = None,
        target_pad_value: int = -100,
        rng_seed: Optional[int] = None,
        max_subtoken_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._language = language
        self._target_pad_value = target_pad_value
        # If not explicitly set, make the inference.
        self._is_continuous = is_continuous if (is_continuous is not None) else (language in {"zh", "ja", "my"})
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
            "punc_pre_target_ids": NeuralType(("B", "T",), LabelsType()),  # D == num_pre_punct_tokens
            "punc_post_target_ids": NeuralType(("B", "T", "D"), LabelsType()),  # D == max_subtoken_len
            "cap_target_ids": NeuralType(("B", "T", "D"), LabelsType()),  # D == max_subtoken_len
            "seg_target_ids": NeuralType(("B", "T"), LabelsType()),
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

    def _select_pre_punct_token_targets(self, tokens: List[str], char_level_targets: List[int]) -> List[int]:
        all_targets: List[int] = []
        # For each token, make one output list
        char_index = 0
        for token in tokens:
            # Keep only the target for the first character of this subword
            all_targets.append(char_level_targets[char_index])
            # Fast-forward to skip the rest of this tokens characters
            num_chars_in_token = len(token)
            if self._using_sp:
                if token.startswith("▁"):
                    num_chars_in_token -= 1
            elif token.startswith("##"):
                num_chars_in_token -= 2
            char_index += num_chars_in_token
        return all_targets

    @typecheck()
    def collate_fn(self, batch):
        inputs = [x[0] for x in batch]
        punct_pre_targets_list = [x[1] for x in batch]
        punct_post_targets_list = [x[2] for x in batch]
        cap_targets_list = [x[3] for x in batch]
        seg_targets_list = [x[4] for x in batch]
        lengths = torch.tensor([x.shape[-1] for x in inputs])
        batch_size = len(inputs)  # should be all the same size

        # Create empty input ID tensors and fill non-padded regions
        input_ids = torch.full(size=(batch_size, lengths.max()), fill_value=self._tokenizer.pad_id)  # noqa
        for i in range(batch_size):
            input_ids[i, : lengths[i]] = inputs[i]

        # Create empty target tensors and fill non-padded regions
        punct_pre_targets = torch.full(size=[batch_size, lengths.max()], fill_value=self._target_pad_value)
        punct_post_targets = torch.full(
            size=[batch_size, lengths.max(), self._max_token_len], fill_value=self._target_pad_value
        )
        cap_targets = torch.full(
            size=[batch_size, lengths.max(), self._max_token_len], fill_value=self._target_pad_value
        )
        seg_targets = torch.full(size=[batch_size, lengths.max()], fill_value=self._target_pad_value)
        for i in range(batch_size):
            cap_targets[i, : lengths[i], :] = cap_targets_list[i]
            seg_targets[i, : lengths[i]] = seg_targets_list[i]
            punct_post_targets[i, : lengths[i], :] = punct_post_targets_list[i]
            punct_pre_targets[i, : lengths[i]] = punct_pre_targets_list[i]

        return input_ids, punct_pre_targets, punct_post_targets, cap_targets, seg_targets, lengths


class PuncTargetsGenerator(abc.ABC):
    """Base class for a punctuation targets generator.

    Base class for generating punctuation targets. Implementations may be language-specific, notably Spanish which uses
    inverted tokens.

    Args:
        post_labels: Punctuation labels that can appear after subwords.
        pre_labels: Punctuation labels that can appear before subwords.
        null_label: The string value of the "null" label, or the label that means "no punctuation here".
    """

    def __init__(
        self, post_labels: List[str], pre_labels: List[str], null_label: str = "<NULL>", ignore_index: int = -100,
    ) -> None:
        self._null_label = null_label
        self._ignore_index = ignore_index

        self._pre_label_to_index = {label: i for i, label in enumerate(pre_labels)}
        self._post_label_to_index = {label: i for i, label in enumerate(post_labels)}
        self._pre_null_index = self._pre_label_to_index[null_label]
        self._post_null_index = self._post_label_to_index[null_label]
        # Save as set for quick membership check
        self._pre_labels = set(pre_labels)
        self._post_labels = set(post_labels)
        self._max_token_len = None

    @abc.abstractmethod
    def generate_targets(self, input_text: str) -> Tuple[str, List[int], List[int]]:
        """Applies punctuation dropout and generates an example.

        Args:
            input_text: Text to process.

        Returns:
            (out_text, pre_targets, post_targets) where `out_text` is the de-punctuated text, and `pre_targets` and
                each contain the target for each non-whitespace character in `new_text`.
        """
        raise NotImplementedError()

    @classmethod
    def from_lang_code(cls, lang_code: str, pre_labels: List[str], post_labels: List[str]):
        """Instantiates a derived class which is applicable to the given language.

        This is a convenience function for instantiating a derived class for a particular language.

        Args:
            lang_code: The language code to use to determine which class to instantiate.
            pre_labels: Punctuation tokens that can appear before a subword.
            post_labels: Punctuation tokens that can appear after a subword.

        """
        lang_code = lang_code.lower()
        if len(lang_code) < 2 or len(lang_code) > 3:
            raise ValueError(f"Only 2- or 3-char lang codes recognized. Got '{lang_code}'.")
        # Catch all the special languages, and default to the English-like punctuation processor.
        if lang_code in {"es", "ast"}:
            # Spanish and Asturian use inverted ?!
            return SpanishPuncTargetsGenerator(pre_labels=pre_labels, post_labels=post_labels)
        elif lang_code in {"zh", "ja", "my"}:
            # Continuous-script languages. The "basic" class seems to work, so nothing special is implemented yet.
            return BasicPuncTargetsGenerator(pre_labels=pre_labels, post_labels=post_labels)
        elif lang_code in {"th"}:
            # Thai -- uses space as punctuation. Don't have a solution, yet.
            raise ValueError(f"Language not supported: {lang_code}")
        else:
            # Assume all other languages use English-like punctuation rules.
            return BasicPuncTargetsGenerator(pre_labels=pre_labels, post_labels=post_labels)


class BasicPuncTargetsGenerator(PuncTargetsGenerator):
    """Punctuation example generator suitable for most languages, including English.

    This class assumes that punctuation tokens appear only after subwords, and will work for most languages.

    """

    def generate_targets(self, input_text: str) -> Tuple[str, List[int], List[int]]:
        # Normalize whitespaces
        input_text = re.sub(r"\s+", " ", input_text)
        # Empty outputs
        out_chars: List[str] = []
        post_targets: List[int] = []
        for input_char in input_text:
            # No targets for spaces because they are ignored when generating subtokens
            if input_char == " ":
                out_chars.append(" ")
                continue
            # Either create a target, or append to the input
            if post_targets and input_char in self._post_labels:
                post_targets[-1] = self._post_label_to_index[input_char]
            else:
                out_chars.append(input_char)
                post_targets.append(self._post_null_index)
        pre_targets = [self._pre_null_index] * len(post_targets)
        out_text = "".join(out_chars)
        return out_text, pre_targets, post_targets


class SpanishPuncTargetsGenerator(PuncTargetsGenerator):
    """Punctuation example generator for Spanish and Asturian.

    """

    def generate_targets(self, input_text: str) -> Tuple[str, List[int], List[int]]:
        # Normalize whitespaces
        input_text = re.sub(r"\s+", " ", input_text)
        # Empty outputs
        out_chars: List[str] = []
        post_targets: List[int] = []
        pre_targets: List[int] = []
        non_whitespace_idx = 0
        for input_char in input_text:
            # Ignore spaces because they are ignored when generating subtokens
            if input_char == " ":
                out_chars.append(" ")
                continue
            # Either create a target, or append to the input
            if input_char in self._post_labels:
                post_targets[-1] = self._post_label_to_index[input_char]
            elif input_char in self._pre_labels:
                pre_targets.append(self._pre_label_to_index[input_char])
            else:
                non_whitespace_idx += 1
                out_chars.append(input_char)
                post_targets.append(self._post_null_index)
                if len(pre_targets) < non_whitespace_idx:
                    pre_targets.append(self._pre_null_index)
        out_text = "".join(out_chars)
        return out_text, pre_targets, post_targets


class CapTargetsGenerator:
    """Generator of true-casing examples.

    Args:
    """

    def __init__(self, ignore_idx: int = -100) -> None:
        self._ignore_idx = ignore_idx

    def _char_is_uncased(self, char: str):
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


class TextPunctCapSegDataset(PunctCapSegDataset):
    """Punctuation, true-casing, and sentence-boundary detection dataset that uses text files for example generation.

    Args:
        text_files: One or more plain-text files with one sentence per line. Each line should be properly true-cased
            and punctuated.
        language: Language code for this dataset.
        punct_pre_labels: List of punctuation tokens that can appear before subwords.
        punct_post_labels: List of punctuation tokens that can appear after subwords.
        tokenizer: TokenizerSpec to use to tokenize the data. Can be set later, for NLP models with forced
            initialization order.
        null_label: The string value of the "null" token, or the token that means "no punctuation here".
        max_length: Maximum length of any input.
        max_lines_per_eg: Uniformly choose between 1 and this many lines to use per example.
        prob_truncate: Truncate examples with this probability.
        truncate_max_tokens: If truncating an example, truncate between 1 and this many tokens.
        target_pad_value: Padding value used in the targets. Should be the ignore_idx of your loss function.
        rng_seed: Seed for the PRNG. For training, keep at None to prevent the data loader works from using the same
            extra indices each step.
    """

    def __init__(
        self,
        text_files: Union[str, List[str]],
        language: str,
        punct_pre_labels: List[str],
        punct_post_labels: List[str],
        is_continuous: Optional[bool] = None,
        tokenizer: Optional[TokenizerSpec] = None,
        null_label: str = "<NULL>",
        max_length: int = 512,
        min_lines_per_eg: int = 1,
        max_lines_per_eg: int = 4,
        prob_truncate: float = 0.2,
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
            is_continuous=is_continuous,
        )
        self._text_files = [text_files] if isinstance(text_files, str) else text_files
        self._null_label = null_label
        self._max_length = max_length
        self._punct_pre_labels = punct_pre_labels
        self._punct_post_labels = punct_post_labels
        self._max_lines_per_eg = max_lines_per_eg
        self._min_lines_per_eg = min_lines_per_eg
        self._prob_truncate = prob_truncate
        self._truncate_max_tokens = truncate_max_tokens
        self._truncate_percentage = truncate_percentage

        self._data: List[str] = self._load_data(self._text_files, max_input_lines)

        self._punct_targets_gen: PuncTargetsGenerator = PuncTargetsGenerator.from_lang_code(
            lang_code=self._language, pre_labels=self._punct_pre_labels, post_labels=self._punct_post_labels,
        )

        self._cap_targets_gen: CapTargetsGenerator = CapTargetsGenerator()

    def _load_data(self, text_files, max_input_lines: Optional[int] = None) -> List[str]:
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
        punctuated_texts: List[str] = [self._data[x] for x in indices_to_use]

        unpunctuated_texts = []
        punct_pre_target_indices = []
        punct_post_target_indices = []
        for text in punctuated_texts:
            unpunct_text, pre_targets, post_targets = self._punct_targets_gen.generate_targets(text)
            unpunctuated_texts.append(unpunct_text)
            punct_pre_target_indices.extend(pre_targets)
            punct_post_target_indices.extend(post_targets)

        # Concatenate all the texts
        concat_unpunct_text = ("" if self._is_continuous else " ").join(unpunctuated_texts)

        # Generate true-case targets and re-case the text
        uncased_text, cap_target_indices = self._cap_targets_gen.generate_targets(concat_unpunct_text)

        # Generate tokens
        input_tokens = self.tokenizer.text_to_tokens(uncased_text)

        # Figure out which characters are the sentence boundaries
        boundary_char_indices: List[int] = []
        for text in unpunctuated_texts:
            num_chars_in_text = len(re.sub(r"\s+", "", text))
            # Subsequent boundaries are in addition to previous
            boundary = num_chars_in_text + (0 if not boundary_char_indices else boundary_char_indices[-1])
            boundary_char_indices.append(boundary)
        char_position = 0
        seg_targets = []
        for token in input_tokens:
            skip = 0
            if self._using_sp:
                if token.startswith("▁"):
                    skip = 1
            elif token.startswith("##"):
                skip = 2
            chars_in_token = len(token) - skip
            char_position += chars_in_token
            # If this token contains the next boundary char, it's a target.
            if boundary_char_indices and char_position >= boundary_char_indices[0]:
                seg_targets.append(1)
                del boundary_char_indices[0]
            else:
                seg_targets.append(0)

        # Finalize the truecase/sentence boundary inputs and targets
        # Fold true-case targets into subword-based
        cap_targets = self._fold_char_targets(input_tokens, cap_target_indices)
        # Trim if too long
        input_ids = self.tokenizer.tokens_to_ids(input_tokens)
        if len(input_ids) + 2 > self._max_length:
            input_ids = input_ids[: self._max_length - 2]
            seg_targets = seg_targets[: self._max_length - 2]
            cap_targets = cap_targets[: self._max_length - 2]
            input_tokens = input_tokens[: self._max_length - 2]
        # # Targeting final token as sentence boundary is not useful. Ignore it.
        # seg_targets[-1] = self._target_pad_value
        # Add BOS/EOS and target padding for those tokens.
        input_ids = [bos] + input_ids + [eos]
        seg_targets = [pad] + seg_targets + [pad]
        cap_targets = pad_list + cap_targets + pad_list

        punct_pre_targets = self._select_pre_punct_token_targets(input_tokens, punct_pre_target_indices)
        punct_pre_targets = [pad] + punct_pre_targets + [pad]
        punct_post_targets = self._fold_char_targets(input_tokens, punct_post_target_indices)
        punct_post_targets = pad_list + punct_post_targets + pad_list

        # Convert to Tensors.
        punct_pre_targets_tensor = torch.tensor(punct_pre_targets, dtype=torch.long)
        punct_post_targets_tensor = torch.tensor(punct_post_targets, dtype=torch.long)
        cap_targets_tensor = torch.tensor(cap_targets, dtype=torch.long)
        seg_targets_tensor = torch.tensor(seg_targets, dtype=torch.long)
        input_ids = torch.tensor(input_ids)
        return (
            input_ids,
            punct_pre_targets_tensor,
            punct_post_targets_tensor,
            cap_targets_tensor,
            seg_targets_tensor,
        )
