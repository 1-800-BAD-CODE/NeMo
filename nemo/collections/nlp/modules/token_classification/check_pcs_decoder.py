from nemo.collections.nlp.modules.token_classification import MHAPunctCapSegDecoder
from nemo.collections.nlp.data.token_classification.punct_cap_seg_dataset import TextPunctCapSegDataset
from nemo.collections.common.tokenizers import SentencePieceTokenizer
import torch
import sys

tokenizer = SentencePieceTokenizer(
    model_path="/Users/shane/corpora/wmt/spe_unigram_64k_lowercase_47lang.model",
    legacy=True,
    special_tokens={"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>"},
)
pre_labels = ["<NULL>", "¿"]
post_labels = [
    "<NULL>",
    ".",
    ",",
    "?",
    "？",
    "，",
    "。",  # Chinese, no enum comma
    "、",
    "・",  # Japanese comma, middle dot
    "।",  # Hindi
    "؟",
    "،",  # Arabic
    ";",  # Greek question mark
    "።",
    "፣",
    "፧",  # Amharic full stop, comma, question
]

dataset = TextPunctCapSegDataset(
    text_files="/Users/shane/corpora/wmt/processed/en.dev.txt",
    tokenizer=tokenizer,
    max_length=128,
    min_lines_per_eg=1,
    max_lines_per_eg=2,
    language="en",
    rng_seed=None,
    punct_pre_labels=pre_labels,
    punct_post_labels=post_labels,
    is_continuous=False,
)

input_ids, punct_pre_targets, punct_post_targets, cap_targets, seg_targets = dataset[10]


encoder_dim = 128

decoder: MHAPunctCapSegDecoder = MHAPunctCapSegDecoder(
    encoder_dim=encoder_dim,
    emb_dim=4,
    punct_num_classes_post=10,
    punct_num_classes_pre=2,
    max_subword_length=16,
    transformer_inner_size=512,
    transformer_num_heads=4,
    transformer_num_layers=2,
    transformer_ffn_dropout=0.1,
    cap_head_dropout=0.1,
    punct_head_dropout=0.1,
    seg_head_dropout=0.1,
    cap_head_n_layers=1,
    punct_head_n_layers=1,
    seg_head_n_layers=1,
)

T = input_ids.size()
encoded = torch.randn(size=[B, T, encoder_dim])
mask = torch.full(size=[B, T], fill_value=1, dtype=torch.int)
punc_mask = punct_post_targets.ne(-100)
print(punc_mask)
# fake batch dim
punc_mask = punc_mask.unsqueeze(0)

punct_logits_pre, punct_logits_post, cap_logits, seg_logits = decoder(encoded=encoded, mask=mask, punc_mask=punc_mask)

print(f"{punct_logits_pre.shape=} {punct_logits_pre=}")
print(f"{punct_logits_post.shape=} {punct_logits_post=}")
print(f"{seg_logits.shape=} {seg_logits=}")
print(f"{cap_logits.shape=} {cap_logits=}")
