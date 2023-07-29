import abc
import numpy as np
import re
import torch
from typing import Dict, List, Optional, Tuple, Union, Set

from nemo.collections.common.tokenizers import TokenizerSpec, SentencePieceTokenizer
from nemo.core import Dataset, typecheck
from nemo.core.neural_types import ChannelType, LengthsType, NeuralType, LabelsType
from nemo.utils import logging

# Define special tokens here, for convenience and consistency
NULL_PUNCT_TOKEN = "<NULL>"
ACRONYM_TOKEN = "<ACRONYM>"


class InferencePunctCapSegDataset(Dataset):
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
        lengths = lengths.add(2).clamp(max=self._max_length)
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
        punct_pre_labels: List[str],
        punct_post_labels: List[str],
        predict_merge: bool = False,
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
        self._punct_pre_labels = punct_pre_labels
        self._punct_post_labels = punct_post_labels
        self._predict_merge = predict_merge
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
        outputs = {
            "input_ids": NeuralType(("B", "T"), ChannelType()),
            "punc_pre_target_ids": NeuralType(("B", "T",), LabelsType()),
            "punc_post_target_ids": NeuralType(("B", "T",), LabelsType()),
            "cap_target_ids": NeuralType(("B", "T", "D"), LabelsType()),  # D == max_subtoken_len
            "seg_target_ids": NeuralType(("B", "T"), LabelsType()),
            "lengths": NeuralType(("B",), LengthsType()),
        }
        if self._predict_merge:
            outputs["merge_target_ids"] = NeuralType(("B", "T"), LabelsType())
        return outputs

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

    def _char_targets_to_token_targets(
        self, tokens: List[str], char_targets: List[int], apply_post: bool = True
    ) -> List[int]:
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
        punct_pre_targets_list = [x[1] for x in batch]
        punct_post_targets_list = [x[2] for x in batch]
        cap_targets_list = [x[3] for x in batch]
        seg_targets_list = [x[4] for x in batch]
        if self._predict_merge:
            merge_targets_list = [x[5] for x in batch]
        lengths = torch.tensor([x.shape[-1] for x in inputs])
        batch_size = len(inputs)  # should be all the same size

        # Create empty input ID tensors and fill non-padded regions
        input_ids = torch.full(size=(batch_size, lengths.max()), fill_value=self._tokenizer.pad_id)  # noqa
        for i in range(batch_size):
            input_ids[i, : lengths[i]] = inputs[i]

        # Create empty target tensors and fill non-padded regions
        punct_pre_targets = torch.full(size=[batch_size, lengths.max()], fill_value=self._target_pad_value)
        punct_post_targets = torch.full(size=[batch_size, lengths.max()], fill_value=self._target_pad_value)
        cap_targets = torch.full(
            size=[batch_size, lengths.max(), self._max_token_len], fill_value=self._target_pad_value
        )
        seg_targets = torch.full(size=[batch_size, lengths.max()], fill_value=self._target_pad_value)
        if self._predict_merge:
            merge_targets = torch.full(size=[batch_size, lengths.max()], fill_value=self._target_pad_value)
        for i in range(batch_size):
            cap_targets[i, : lengths[i], :] = cap_targets_list[i]
            seg_targets[i, : lengths[i]] = seg_targets_list[i]
            punct_post_targets[i, : lengths[i]] = punct_post_targets_list[i]
            punct_pre_targets[i, : lengths[i]] = punct_pre_targets_list[i]
            if self._predict_merge:
                merge_targets[i, : lengths[i]] = merge_targets_list[i]  # noqa

        if self._predict_merge:
            return input_ids, punct_pre_targets, punct_post_targets, cap_targets, seg_targets, lengths, merge_targets
        return input_ids, punct_pre_targets, punct_post_targets, cap_targets, seg_targets, lengths


class MergeTargetsGenerator:
    """Class for a "merge" targets generator.

    Splits acronyms into single-letter tokens such as
        "FBI" -> "F B I"

        "СИАБ." -> "С И А Б."

        "U.S." -> "U. S."

        "a.m." -> "a. m."

        "¿U.S." -> "¿U. S."  (actually misses this case, currently)

    The intent of this class is to allow learning to "merge" spelled-out acronyms which are transcribed by an ASR model
    as individual characters. E.g., a typical ASR model will transcribe "the f b i agent" and in post-processing we want
    to merge this into "the fbi agent".

    Since it is difficult to deduce which tokens will be transcribed as characters, e.g., "f b i", or as contiguous
    tokens, e.g., "nato", we split (with some probability) any token which is all upper-case or contains a period after
    each char (e.g., 'a.m.').

    Args:
        post_labels: Post-punctuation tokens, to be ignored in counting character positions
        pre_labels: Pre-punctuation tokens, analogous to `post_labels`
        p_split: Split acronyms with this probability.
    """

    def __init__(self, post_labels: List[str], pre_labels: List[str], p_split: float = 0.8) -> None:
        self._p_split = p_split
        self._all_punc: Set[str] = set(post_labels + pre_labels)
        self._all_punc.discard(NULL_PUNCT_TOKEN)
        self._all_punc.discard(ACRONYM_TOKEN)
        # The regex we want for splitting is "<optional_punct_token><character><optional_punct_token>"
        punc_str = ''.join(re.escape(x) for x in self._all_punc)
        # Regex used to split an acronym into constituent chars with optional periods. Match zero or one punc tokens on
        # each side of a non-punc character.
        self._split_ptn = re.compile(rf"([{punc_str}]?[^{punc_str}][{punc_str}]?)")
        self._punc_ptn = re.compile(rf"[{punc_str}]+")
        self._rng = np.random.default_rng()  # todo seed? doesn't matter

    def generate_targets(self, input_text: str) -> Tuple[str, List[int]]:
        processed_tokens: List[str] = []
        char_level_targets: List[int] = []
        for token in input_text.split():
            # Split on multi-char uppercase tokens: FBI, U.S., etc. Also catch lower-cased acronyms, a.m., etc.
            #
            # If all non-punc chars in the token belong to cased alphabets and are upper-cased, it's an acronym
            # If all chars in the token are followed by a period, it's an acronym
            # The `if` structure is convoluted, but enables short-circuits around expensive checks.
            do_split = False
            if len(token) > 1 and self._rng.random() < self._p_split:
                # Check if every non-punc token is upper-case, e.g, NATO, FBI, etc.
                token_no_punc = self._punc_ptn.sub("", token)
                # Verify upper char-by-char, because`"A認".isupper() == True` but `"認".isupper() == False`.
                if len(token_no_punc) > 1 and all(x.isupper() for x in token_no_punc):
                    do_split = True
                # Check for punctuated initialisms: a.m., p.m., etc. Candidates must have 4+ chars.
                if not do_split and len(token) >= 4:
                    # This check fails for, e.g., '¿a.m.' but this pattern appears 0 times in news training data
                    every_other_char = token[1::2]
                    do_split = all(x == "." for x in every_other_char)

            if not do_split:
                processed_tokens.append(token)
                for char in token:
                    if char in self._all_punc:
                        # Don't count punctuation since it'll be removed before example is generated
                        continue
                    char_level_targets.append(0)
            else:
                # This is an acronym that we should split
                subtokens = self._split_ptn.findall(token)
                processed_tokens.extend(subtokens)
                # "merge[i]" implies "remove the space between chars i and i+1", e.g.,
                #    "FBI agent" -> (['F', 'B', 'I', 'agent'], [merge, merge, no, no, no, no, no, no])
                # such that we merge after 'F' and 'B' to recover "FBI agent". The reason for using character-level
                # targets is to make it easy to align with subword tokens down-stream.
                for i, subtoken in enumerate(subtokens):
                    target = 1 if i < len(subtokens) - 1 else 0
                    # Each of these tokens should have exactly one non-punc char, as dictated by the regex
                    char_level_targets.append(target)
        out_text = " ".join(processed_tokens)
        return out_text, char_level_targets


class PuncTargetsGenerator(abc.ABC):
    """Base class for a punctuation targets generator.

    Base class for generating punctuation targets. Implementations may be language-specific, notably Spanish which uses
    inverted tokens.

    Args:
        post_labels: Punctuation labels that can appear after subwords.
        pre_labels: Punctuation labels that can appear before subwords.
    """

    def __init__(self, post_labels: List[str], pre_labels: List[str], ignore_index: int = -100,) -> None:
        self._ignore_index = ignore_index

        self._pre_label_to_index = {label: i for i, label in enumerate(pre_labels)}
        self._post_label_to_index = {label: i for i, label in enumerate(post_labels)}
        self._pre_null_index = self._pre_label_to_index[NULL_PUNCT_TOKEN]
        self._post_null_index = self._post_label_to_index[NULL_PUNCT_TOKEN]
        # Save as set for quick membership check
        self._pre_labels = set(pre_labels)
        self._post_labels = set(post_labels)
        self._max_token_len = None
        self._using_acronyms = ACRONYM_TOKEN in post_labels

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
        if lang_code in {"th"}:
            # Thai -- uses space as punctuation. Don't have a solution, yet.
            raise ValueError(f"Language not supported: {lang_code}")
        else:
            # Assume all other languages use English-like punctuation rules.
            # Spanish and Asturian use inverted ?! Basic generator works ok, just be sure there are no "pre" tokens in
            # languages that don't use it.
            # Continuous-script languages ok, too.
            return BasicPuncTargetsGenerator(pre_labels=pre_labels, post_labels=post_labels)


class BasicPuncTargetsGenerator(PuncTargetsGenerator):
    """Punctuation example generator that works for most languages.

    """

    def generate_targets(self, input_text: str) -> Tuple[str, List[int], List[int]]:
        # Normalize whitespaces
        input_text = re.sub(r"\s+", " ", input_text)
        # Empty outputs
        out_chars: List[str] = []
        post_targets: List[int] = []
        pre_targets: List[int] = []
        non_whitespace_idx = 0
        if self._using_acronyms:
            period_or_acronym_index = [self._post_label_to_index["."], self._post_label_to_index[ACRONYM_TOKEN]]
        for input_char in input_text:
            # Ignore spaces because they are ignored when generating subtokens
            if input_char == " ":
                out_chars.append(" ")
                continue
            # Either create a target, or append to the input
            if input_char in self._post_labels:
                # If two consecutive non-space chars end with '.', both are acronyms
                # There's probably a more graceful way to implement this check.
                if (
                    self._using_acronyms
                    and input_char == "."
                    and len(post_targets) > 1
                    and out_chars[-2] != " "
                    and post_targets[-2] in period_or_acronym_index  # noqa
                ):
                    post_targets[-2] = self._post_label_to_index[ACRONYM_TOKEN]
                    post_targets[-1] = self._post_label_to_index[ACRONYM_TOKEN]
                else:
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
        max_length: Maximum length of any input.
        max_lines_per_eg: Uniformly choose between 1 and this many lines to use per example.
        prob_truncate: Truncate examples with this probability.
        truncate_max_tokens: If truncating an example, truncate between 1 and this many tokens.
        target_pad_value: Padding value used in the targets. Should be the ignore_idx of your loss function.
        rng_seed: Seed for the PRNG. For training, keep at None to prevent the data loader works from using the same
            extra indices each step.
        predict_merge: Whether to make "merge" predictions, i.e., whether to remove the whitespace between token `i` and
            token `i+1`.
    """

    def __init__(
        self,
        text_files: Union[str, List[str]],
        language: str,
        punct_pre_labels: List[str],
        punct_post_labels: List[str],
        is_continuous: Optional[bool] = None,
        tokenizer: Optional[TokenizerSpec] = None,
        max_length: int = 512,
        min_lines_per_eg: int = 1,
        max_lines_per_eg: int = 4,
        prob_truncate: float = 0.0,
        truncate_max_tokens: int = 5,
        truncate_percentage: float = 0.0,
        target_pad_value: int = -100,
        rng_seed: Optional[int] = None,
        max_input_lines: Optional[int] = None,
        predict_merge: bool = False,
    ):
        super().__init__(
            language=language,
            punct_post_labels=punct_post_labels,
            punct_pre_labels=punct_pre_labels,
            tokenizer=tokenizer,
            target_pad_value=target_pad_value,
            rng_seed=rng_seed,
            is_continuous=is_continuous,
            predict_merge=predict_merge,
        )
        self._text_files = [text_files] if isinstance(text_files, str) else text_files
        self._max_length = max_length
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

        self._predict_merge = predict_merge
        self._merge_targets_gen: Optional[MergeTargetsGenerator] = None
        if predict_merge:
            self._merge_targets_gen = MergeTargetsGenerator(
                post_labels=self._punct_post_labels, pre_labels=self._punct_pre_labels
            )

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

        # Randomly choose how many additional lines to use. During testing, we may fix the values
        if self._min_lines_per_eg == self._max_lines_per_eg:
            num_extra_lines = self._min_lines_per_eg - 1
        else:
            num_extra_lines = self._rng.integers(self._min_lines_per_eg - 1, self._max_lines_per_eg)
        extra_indices = list(self._rng.integers(low=0, high=len(self), size=num_extra_lines))
        # Randomly select additional indices to use
        indices_to_use = [idx] + extra_indices
        punctuated_texts: List[str] = [self._data[x] for x in indices_to_use]

        # Split acronyms in text; get character-level targets
        merge_target_indices: List[int] = []
        if self._predict_merge:
            for i, text in enumerate(punctuated_texts):
                text, targets = self._merge_targets_gen.generate_targets(text)
                punctuated_texts[i] = text
                merge_target_indices.extend(targets)

        # Strip punctuation and get character-level punctuation targets
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
        # Trim if too long
        cap_targets = self._fold_char_targets(input_tokens, cap_target_indices)
        input_ids = self.tokenizer.tokens_to_ids(input_tokens)
        if len(input_ids) + 2 > self._max_length:
            input_ids = input_ids[: self._max_length - 2]
            seg_targets = seg_targets[: self._max_length - 2]
            cap_targets = cap_targets[: self._max_length - 2]
            input_tokens = input_tokens[: self._max_length - 2]
        if self._predict_merge:
            merge_targets = self._char_targets_to_token_targets(input_tokens, merge_target_indices)
            if len(merge_targets) + 2 > self._max_length:
                merge_targets = merge_targets[: self._max_length - 2]
            merge_targets = [pad] + merge_targets + [pad]

        # # Targeting final token as sentence boundary is not useful. Ignore it.
        # seg_targets[-1] = self._target_pad_value
        # Add BOS/EOS and target padding for those tokens.
        input_ids = [bos] + input_ids + [eos]
        seg_targets = [pad] + seg_targets + [pad]
        cap_targets = pad_list + cap_targets + pad_list

        punct_pre_targets = self._char_targets_to_token_targets(
            input_tokens, punct_pre_target_indices, apply_post=False
        )
        punct_pre_targets = [pad] + punct_pre_targets + [pad]
        punct_post_targets = self._char_targets_to_token_targets(
            input_tokens, punct_post_target_indices, apply_post=True
        )
        punct_post_targets = [pad] + punct_post_targets + [pad]

        # Convert to Tensors.
        punct_pre_targets_tensor = torch.tensor(punct_pre_targets, dtype=torch.long)
        punct_post_targets_tensor = torch.tensor(punct_post_targets, dtype=torch.long)
        cap_targets_tensor = torch.tensor(cap_targets, dtype=torch.long)
        seg_targets_tensor = torch.tensor(seg_targets, dtype=torch.long)
        input_ids = torch.tensor(input_ids)
        if self._predict_merge:
            merge_targets_tensor = torch.tensor(merge_targets, dtype=torch.long)
            return (
                input_ids,
                punct_pre_targets_tensor,
                punct_post_targets_tensor,
                cap_targets_tensor,
                seg_targets_tensor,
                merge_targets_tensor,
            )
        else:
            return (
                input_ids,
                punct_pre_targets_tensor,
                punct_post_targets_tensor,
                cap_targets_tensor,
                seg_targets_tensor,
            )
