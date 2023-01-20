import torch

from nemo.collections.nlp.modules.token_classification import MHAPunctCapSegDecoder

encoder_dim = 128

decoder: MHAPunctCapSegDecoder = MHAPunctCapSegDecoder(
    encoder_dim=encoder_dim,
    punct_emb_dim=4,
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

encoded = torch.randn(size=[3, 44, encoder_dim])
mask = torch.full(size=[3, 44], fill_value=1, dtype=torch.int)

punct_logits_pre, punct_logits_post, cap_logits, seg_logits = decoder(encoded=encoded, mask=mask)

print(f"{punct_logits_pre.shape=} {punct_logits_pre=}")
print(f"{punct_logits_post.shape=} {punct_logits_post=}")
print(f"{seg_logits.shape=} {seg_logits=}")
print(f"{cap_logits.shape=} {cap_logits=}")
