

import argparse
import os
import sys
import abc
import random
import re
import unicodedata
from unicodedata import category
from dataclasses import dataclass, field
from typing import List, Optional

from tqdm import tqdm
import numpy as np
import hydra
import opencc
from omegaconf import OmegaConf, MISSING
from num2words import num2words
from nemo.utils import logging
from joblib import Parallel, delayed
from sentencepiece import SentencePieceProcessor


class TextCleaner(abc.ABC):
    """Base class for language-specific text cleaners.
    The classes derived from this base class will be applied to each input sentence before we generate examples. The
    main idea is that these classes normalize the data specific to our task.
    """

    @abc.abstractmethod
    def clean(self, text: str) -> str:
        raise NotImplementedError()


@dataclass
class DatasetConfig:
    text_files: List[str] = MISSING
    text_cleaners: List[TextCleaner] = field(default_factory=lambda: [])
    language: str = MISSING
    continuous_script: bool = False
    ignore_case: bool = False


@dataclass
class MultiDatasetConfig:
    datasets: List[DatasetConfig] = MISSING
    output_dir: str = MISSING
    punct_post_labels: List[str] = MISSING
    # field(
    #     default_factory=lambda: [".", ",", "?", "？", "，", "。", "、", "・", "।", "؟", "،", "՞", ";", "።",]
    # )
    max_words_per_line: int = 20
    min_words_per_line: int = 1
    max_chars_per_line: int = 24
    min_chars_per_line: int = 2
    dev_set_num_lines: int = 2000
    test_set_num_lines: int = 2000
    punct_pre_labels: List[str] = field(default_factory=lambda: ["¿"])
    # Note: semi-colon is for Greek. Remove it from other languages.
    max_lines_per_input_file: int = 1000000
    max_lines_total: int = 1000000


class SentencePieceNormalizer(TextCleaner):
    def __init__(
        self,
        sp_model_path: str,
        remove_oovs: bool = True
    ):
        self._sp_model: SentencePieceProcessor = SentencePieceProcessor(sp_model_path)
        self._remove_oovs = remove_oovs

    def clean(self, text: str) -> str:
        ids = self._sp_model.EncodeAsIds(text)
        if self._remove_oovs and self._sp_model.unk_id() in ids:
            return ""
        text = self._sp_model.Decode(ids)
        return text


class CharFilter(TextCleaner):
    """Removes all unwanted punctuation
    """

    def __init__(
        self,
        post_punct_labels: List[str],
        pre_punct_labels: List[str],
        extra_chars_to_keep: Optional[List[str]] = None,
        no_punct_in_numbers: bool = True,
        remove_semicolon: bool = True,
        remove_if_no_end_punc: bool = True
    ):
        self._remove_if_no_end_punc = remove_if_no_end_punc
        self._post_labels = set(post_punct_labels)
        if extra_chars_to_keep is None:
            extra_chars_to_keep = []
        self._punct_to_remove = set(
            chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')
        )
        punct_to_keep = set(post_punct_labels + pre_punct_labels + extra_chars_to_keep)
        # Always keep apostrophes
        punct_to_keep.add("'")
        for c in punct_to_keep:
            self._punct_to_remove.remove(c)
        # When using Greek, ; is a valid punct token but should be removed from non-Greek.
        if remove_semicolon:
            self._punct_to_remove.add(";")
        self._hyphen_ptn = re.compile(r"-")
        # Seems to be the same speed to use a regex or a character filter
        # all_chars_str = "".join([re.escape(x) for x in all_punct])
        # self._remove_char_ptn = re.compile(f"[{all_chars_str}]+")
        self._no_punct_in_numbers = no_punct_in_numbers
        self._punct_in_numbers_ptn = re.compile(r"([0-9]+)[\.,]+([0-9]+)")
        # Don't really need to find all of them
        self._url_ptn = re.compile(r"\.(com|org|gov|edu)")
        # Remove apostrophes if not within a word
        self._remove_apostrophe_ptn = re.compile(r"(\B'|'\B)")

    def clean(self, text: str) -> str:
        # If there is a website in this string, skip the whole line. Easiest way to handle it
        if self._url_ptn.search(text) is not None:
            return ""
        # Replace hyphens with a space
        text = self._hyphen_ptn.sub(" ", text)
        # text = self._remove_char_ptn.sub("", text)
        text = "".join(x for x in text if x not in self._punct_to_remove)
        if self._no_punct_in_numbers:
            text = self._punct_in_numbers_ptn.sub(r"\g<1>\g<2>", text)
        text = self._remove_apostrophe_ptn.sub("", text)
        text = text.strip()
        if text and self._remove_if_no_end_punc and text[-1] not in self._post_labels:
            return ""
        return text


class StandardPunctNormalizer(TextCleaner):
    """Class for normalizing punctuation in most languages.
    Intended to be run on plain text data before generating examples.
    First, removes all spaces that appear before punctuation tokens. e.g.,
    "foo ." -> "foo.", "foo. . ." -> "foo...", etc.
    Then replaces all instances of 2+ consecutive punctuation tokens by the first punctuation token in the sequence.
    E.g.,
    "foo..." -> "foo."
    Note on the latter that this primarily deals with 1) ellipsis and 2) messy data. In the former case, we replace
    ellipsis with a period, in that latter we do the best we can with messy data in a simple way.
    Args:
        punct_tokens: List of punctuation tokens.
    """

    def __init__(self, punct_tokens: List[str], default_full_stop: Optional[str] = None) -> None:
        self._punct_tokens = set(punct_tokens)
        self._default_full_stop = default_full_stop
        # This assumes all punctuation tokens are single characters, which should be true
        escaped_tokens = [re.escape(x) for x in punct_tokens]
        punct_str = "".join(escaped_tokens)
        # Match a punct token, followed immediately by more. Capture the first token.
        self._multi_punct_ptn = re.compile(rf"([{punct_str}])[{punct_str}]+")
        # Match whitespace followed by a punct token. Capture the token.
        self._whitespace_ptn = re.compile(rf"\s+([{punct_str}])")
        # Match punctuation at the beginning of a sentence (not valid except in Spanish)
        self._punct_at_bos_ptn = re.compile(rf"^[{punct_str}\s]+")

    def clean(self, text: str) -> str:
        # Remove punctuation/space at beginning of sentence
        text = self._punct_at_bos_ptn.sub("", text)
        # Remove whitespace before any punctuation token
        text = self._whitespace_ptn.sub(r"\g<1>", text)
        # Replace consecutive punctuation tokens with the first tokens
        text = self._multi_punct_ptn.sub(r"\g<1>", text)
        text = text.strip()
        if self._default_full_stop is not None and text and text[-1] not in self._punct_tokens:
            text = text + self._default_full_stop
        return text


class SpanishPunctNormalizer(TextCleaner):
    """Class for normalizing punctuation Spanish.
    Similar to a :class:``StandardPunctNormalizer`` but has special rules for dealing with "¡" and "¿".
    For non-inverted punctuation, we follow the same rules as :class:``StandardPunctNormalizer``.
    For inverted punctuation, we allow them to appear at the beginning of a string and allow space before them (but not
    after).
    Args:
        pre_punct_tokens: List of punctuation tokens that can appear before a subword. Basically, inverted punctuation.
        post_punct_tokens: List of punctuation tokens that can appear after a subword.
    """

    def __init__(self, pre_punct_tokens: List[str], post_punct_tokens: List[str], default_full_stop: Optional[str] = None) -> None:
        self._default_full_stop = default_full_stop
        self._punct_tokens = set(post_punct_tokens + pre_punct_tokens)
        # make char classes e.g. '[\.,?]'
        post_punct_char_str = "".join([re.escape(x) for x in post_punct_tokens])
        pre_punct_char_str = "".join([re.escape(x) for x in pre_punct_tokens])
        all_char_str = "".join([re.escape(x) for x in pre_punct_tokens + post_punct_tokens])

        # Match whitespace followed by a non-inverted token. Capture the token.
        self._whitespace_ptn1 = re.compile(rf"\s+([{post_punct_char_str}])")
        # Match whitespace after inverted token. Capture the token.
        self._whitespace_ptn2 = re.compile(rf"([{pre_punct_char_str}])\s+")

        # Catch inverted punctuation at eos
        self._pre_punct_at_eos_ptn = re.compile(rf"[{pre_punct_char_str}\s]+$")
        # Catch non-inverted at bos
        self._post_punct_at_bos_ptn = re.compile(rf"^[{post_punct_char_str}\s]+")
        # Catch inverted followed by any punctuation, replace with inverted
        self._multi_punct_ptn1 = re.compile(rf"([{pre_punct_char_str}])[{all_char_str}]+")
        # Catch non-inverted followed by any tokens without space
        self._multi_punct_ptn2 = re.compile(rf"([{post_punct_char_str}])[{all_char_str}]+")

    def clean(self, text: str) -> str:
        # Remove punctuation/space at beginning of sentence
        text = self._post_punct_at_bos_ptn.sub("", text)
        text = self._pre_punct_at_eos_ptn.sub("", text)
        # Remove whitespace before any punctuation token
        text = self._whitespace_ptn1.sub(r"\g<1>", text)
        text = self._whitespace_ptn2.sub(r"\g<1>", text)
        # Replace consecutive punctuation tokens with the first tokens
        text = self._multi_punct_ptn1.sub(r"\g<1>", text)
        text = self._multi_punct_ptn2.sub(r"\g<1>", text)
        if self._default_full_stop is not None and text and text[-1] not in self._punct_tokens:
            text = text + self._default_full_stop
        return text


class ChineseTextCleaner(TextCleaner):
    """Text cleaner for Chinese.
    Args:
        remove_spaces: If True, remove all spaces from the text.
        replace_latin: If true, replace all instances of latin punctuation with the analogous Chinese token. E.g.,
            replace all instances of '.' with '。'
        no_enum_comma: If true, replace all instances of the Chinese enumeration comma "、" with the comma ",". Most
            datasets use these commas interchangeably, so unless you are sure your data correctly and consistently uses
            the enumeration comma correctly, you should set this to True. Otherwise the model will be penalized for
            the messy data.
    """

    def __init__(
            self,
            remove_spaces: bool = True,
            replace_latin: bool = True,
            no_enum_comma: bool = True,
            s2t_percent: float = 0.5
    ) -> None:
        self._remove_spaces = remove_spaces
        self._replace_latin = replace_latin
        self._no_enum_comma = no_enum_comma
        self._s2t_percent = s2t_percent
        self._converter = opencc.OpenCC("s2t.json")

    def clean(self, text: str) -> str:
        if self._remove_spaces:
            text = re.sub(r"\s+", "", text)
        if np.random.rand() < self._s2t_percent:
            text = self._converter.convert(text)
        if self._replace_latin:
            # Replace latin punctuation with Chinese.
            text = re.sub(r"\?", "？", text)
            # # Allow latin comma in numbers
            # text = re.sub(r"(?<=\D),(?=\D)", "，", text)
            text = re.sub(r",", "，", text)
            # Only swap periods if they are at the end of a sentence; else assume they are not full stops.
            text = re.sub(r"[\\.．]", "。", text)
            text = re.sub(r"!", "！", text)
        if self._no_enum_comma:
            # Replace the enumeration comma with regular comma. These two are often used interchangeably in
            # raw data, so it is difficult or impossible to learn the enumeration comma.
            text = re.sub(r"、", "，", text)
        # 'interpunct' appears very rarely, it can mess up the validation metrics.
        text = re.sub(r"・", "", text)
        return text


class JapaneseTextCleaner(TextCleaner):
    """Text cleaner for Japanese.
    Args:
        remove_spaces: If True, remove all spaces from the text.
        replace_latin: If true, replace all instances of latin punctuation with the analogous Chinese token. E.g.,
            replace all instances of '.' with '。'
    """

    def __init__(self, remove_spaces: bool = True, replace_latin: bool = True,) -> None:
        self._remove_spaces = remove_spaces
        self._replace_latin = replace_latin

    def clean(self, text: str) -> str:
        if self._remove_spaces:
            text = re.sub(r"\s+", "", text)
        if self._replace_latin:
            # Replace latin punctuation with Chinese.
            text = re.sub(r"\?", "？", text)
            text = re.sub(r"!", "！", text)
            # # Allow latin comma within numbers
            # text = re.sub(r"(?<=\D),(?=\D)", "，", text)
            text = re.sub(r",", "，", text)
            text = re.sub(r"[\\.．]", "。", text)
        # The full-width comma is too rare, even when used correctly.
        text = re.sub(r"，", "，", text)
        return text


class ArabicTextCleaner(TextCleaner):
    """Text cleaner for Arabic.
    Args:
        replace_latin: If true, replace all instances of latin punctuation with the analogous Arabic token. E.g.,
            replace all instances of '?' with '؟'
    """

    def __init__(self, replace_latin: bool = True,) -> None:
        self._replace_latin = replace_latin

    def clean(self, text: str) -> str:
        if self._replace_latin:
            # Replace latin punctuation with Arabic equivalent (reversed '?' and ',').
            text = re.sub(r"\?", "؟", text)
            text = re.sub(r",", "،", text)
        return text


class HindiTextCleaner(TextCleaner):
    def __init__(self, no_double_danda: bool = True, replace_latin: bool = True,) -> None:
        self._no_double_danda = no_double_danda
        self._replace_latin = replace_latin

    def clean(self, text: str) -> str:
        text = text.strip()
        if self._no_double_danda:
            text = re.sub(r"॥", "।", text)
        if self._replace_latin:
            text = re.sub(r"\.", "।", text)
        return text


class GreekTextCleaner(TextCleaner):
    def __init__(self, replace_latin_question_mark: bool = True,) -> None:
        self._replace_latin_question_mark = replace_latin_question_mark
        self._latin_question_mark_ptn = re.compile(r"\?")

    def clean(self, text: str) -> str:
        if self._replace_latin_question_mark:
            text = self._latin_question_mark_ptn.sub(";", text)
        return text


class AmharicTextCleaner(TextCleaner):
    def __init__(self,) -> None:
        # If a sentence ends with a period, change it to a four dots
        self._period_ptn = re.compile(r"\.")
        # Sometimes two colons ("::") are used instead of the four dots
        self._two_colon_ptn = re.compile("::")
        self._latin_comma_ptn = re.compile(",")
        self._latin_question_ptn = re.compile(r"\?")

    def clean(self, text: str) -> str:
        text = self._period_ptn.sub("።", text)
        text = self._two_colon_ptn.sub("።", text)
        text = self._latin_comma_ptn.sub("፣", text)
        text = self._latin_question_ptn.sub("፧", text)
        return text


class GujaratiTextCleaner(TextCleaner):
    def __init__(self,) -> None:
        # Per wikipedia, "Contemporary Gujarati uses English punctuation"
        self._danda_ptn = re.compile(r"।")

    def clean(self, text: str) -> str:
        text = self._danda_ptn.sub(".", text)
        return text


def _process_dataset(cfg: MultiDatasetConfig, dataset_index: int):
    all_punct_labels = set(cfg.punct_post_labels + cfg.punct_pre_labels)
    # Make a regex to determine whether some line contains nothing but punctuation.
    joined_punct_tokens = "".join([re.escape(x) for x in all_punct_labels])
    all_punct_ptn = re.compile(rf"^[{joined_punct_tokens}\s]*$")
    bos_delete_ptn = re.compile(rf"^[\s—-]+")
    dataset = cfg.datasets[dataset_index]
    cleaners = [hydra.utils.instantiate(x) for x in dataset.text_cleaners]
    all_lines = []
    is_continous = dataset.get("continuous_script", False)
    ignore_case = dataset.get("ignore_case", False)
    for text_file in dataset.text_files:
        logging.info(f"Processing {text_file}")
        with open(text_file) as f, tqdm(total=cfg.max_lines_per_input_file) as pbar:
            lines_scanned = 0
            num_lines_from_file = 0
            for line in f:
                if num_lines_from_file >= cfg.max_lines_per_input_file or len(all_lines) > cfg.max_lines_total:
                    break
                lines_scanned += 1
                line = line.strip()
                line = bos_delete_ptn.sub("", line)
                if not line:
                    continue
                # Drop if line does not start with an upper-case letter.
                # Note: for uncase chars, islower() == isupper() == False, so no action is taken.
                if not ignore_case and line[0].islower():
                    continue
                # Apply all preprocessors
                for cleaner in cleaners:
                    line = cleaner.clean(line)
                if not line:
                    continue
                # Drop if too short/long
                if is_continous:
                    # Continuous langs, check only char counts
                    num_chars = len(line)
                    if num_chars > cfg.max_chars_per_line:
                        continue
                    if num_chars < cfg.min_chars_per_line:
                        continue
                else:
                    num_words = len(line.split())
                    # Drop if line contains too many words, if specified.
                    if num_words > cfg.max_words_per_line:
                        continue
                    if num_words < cfg.min_words_per_line:
                        continue
                # Drop is entire sentence is upper case
                if not ignore_case and line.isupper():
                    continue
                # Drop if just punctuation
                if not line or all_punct_ptn.match(line):
                    continue
                # Normalize space, just to make it easier to inspect
                line = re.sub(r"\s+", " ", line)
                all_lines.append(line)
                pbar.update(1)
                num_lines_from_file += 1
        pbar.close()
        lines_skipped = lines_scanned - num_lines_from_file
        skip_ratio = lines_skipped / lines_scanned if lines_scanned > 0 else 0.0
        logging.info(
            f"Dataset for '{dataset.language}' collected {num_lines_from_file} lines from '{text_file}'; skipped "
            f"{lines_skipped} ({skip_ratio:0.2%}) from this file."
        )

    train_start, train_stop = 0, len(all_lines) - cfg.dev_set_num_lines - cfg.test_set_num_lines
    test_start, test_stop = train_stop, train_stop + cfg.test_set_num_lines
    with open(f"{cfg.output_dir}/{dataset.language}.train.txt", "w") as writer:
        for line in all_lines[train_start:train_stop]:
            writer.write(f"{line}\n")
    with open(f"{cfg.output_dir}/{dataset.language}.test.txt", "w") as writer:
        for line in all_lines[test_start:test_stop]:
            writer.write(f"{line}\n")
    with open(f"{cfg.output_dir}/{dataset.language}.dev.txt", "w") as writer:
        for line in all_lines[test_stop:]:
            writer.write(f"{line}\n")

    logging.info(
        f"Dataset for '{dataset.language}' collected {len(all_lines)} lines from {len(dataset.text_files)} "
        f"file{'s' if len(dataset.text_files) > 1 else ''}."
    )


def main():
    # Note `hydra_runner` is not used because we are not in a Python package here, and there were issues initializing
    # classes of the form `__main__.ClassName`. The alternative used here is a messy compromise.
    cfg: MultiDatasetConfig = OmegaConf.load("data.yaml")
    os.makedirs(cfg.output_dir, exist_ok=True)
    Parallel(n_jobs=6, backend="multiprocessing")(
        delayed(_process_dataset)(
            cfg=cfg,
            dataset_index=i
        ) for i in range(len(cfg.datasets))
    )


if __name__ == "__main__":
    main()
