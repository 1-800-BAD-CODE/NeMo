import abc
from typing import Optional, Dict

import torch
import torch.nn as nn

from nemo.collections.nlp.modules.common.transformer import TransformerEncoder
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.core import typecheck
from nemo.core.classes import NeuralModule
from nemo.core.neural_types import NeuralType, ChannelType, LogitsType, MaskType, LabelsType
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector


class PunctCapSegDecoder(NeuralModule):
    def __init__(self,):
        super().__init__()

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        When training, the reference punctuation targets are used to condition the true-casing and sentence-boundary
        heads.

        During inference, the model uses the punctuation predictions. In this case, we need a mask to know the length
        of each subword, to ignore padded character positions within the punctuation predictions.
        """
        return {
            "encoded": NeuralType(("B", "T", "D"), ChannelType()),
            "mask": NeuralType(("B", "T"), MaskType()),
            "punc_targets": NeuralType(("B", "T", "C"), LabelsType(), optional=True),  # training - reference targets
            "seg_targets": NeuralType(("B", "T"), LabelsType(), optional=True),  # training - reference targets
            "punc_mask": NeuralType(("B", "T", "C"), MaskType(), optional=True),  # inference - valid characters
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "punct_pre_logits": NeuralType(("B", "T", "C"), LogitsType()),
            "punct_post_logits": NeuralType(("B", "T", "D", "C"), LogitsType()),  # D == max_subword_length
            "cap_logits": NeuralType(("B", "T", "D"), LogitsType()),  # D == max_subword_length
            "seg_logits": NeuralType(("B", "T", "C"), LogitsType()),  # C == 2
        }

    @abc.abstractmethod
    def forward(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        punc_mask: Optional[torch.Tensor] = None,
        punc_targets: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError()

    @classmethod
    def restore_from(
        cls,
        restore_path: str,
        override_config_path: Optional[str] = None,
        map_location: Optional["torch.device"] = None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Optional["Trainer"] = None,  # noqa
        save_restore_connector: SaveRestoreConnector = None,
    ):
        pass

    def save_to(self, save_path: str):
        pass


class LinearPunctCapSegDecoder(PunctCapSegDecoder):
    """


    """

    def __init__(
        self,
        encoder_dim: int,
        punct_num_classes_post: int,
        punct_num_classes_pre: int,
        max_subword_length: int,
        punct_head_n_layers: int = 1,
        punct_head_dropout: float = 0.1,
        cap_head_n_layers: int = 1,
        cap_head_dropout: float = 0.1,
        seg_head_n_layers: int = 1,
        seg_head_dropout: float = 0.1,
    ):
        super().__init__()
        self._max_subword_len = max_subword_length
        self._num_post_punct_classes = punct_num_classes_post
        self._num_pre_punct_classes = punct_num_classes_pre
        self._punct_head_post: TokenClassifier = TokenClassifier(
            hidden_size=encoder_dim,
            num_layers=punct_head_n_layers,
            dropout=punct_head_dropout,
            activation="relu",
            log_softmax=False,
            num_classes=punct_num_classes_post * max_subword_length,
        )
        self._punct_head_pre: TokenClassifier = TokenClassifier(
            hidden_size=encoder_dim,
            num_layers=punct_head_n_layers,
            dropout=punct_head_dropout,
            activation="relu",
            log_softmax=False,
            num_classes=punct_num_classes_pre,
        )
        self._seg_head: TokenClassifier = TokenClassifier(
            hidden_size=encoder_dim,
            num_layers=seg_head_n_layers,
            dropout=seg_head_dropout,
            activation="relu",
            log_softmax=False,
            num_classes=2,
        )
        self._cap_head: TokenClassifier = TokenClassifier(
            hidden_size=encoder_dim,
            num_layers=cap_head_n_layers,
            dropout=cap_head_dropout,
            activation="relu",
            log_softmax=False,
            num_classes=max_subword_length,
        )

    @typecheck()
    def forward(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        punc_mask: Optional[torch.Tensor] = None,
        punc_targets: Optional[torch.Tensor] = None,
        seg_targets: Optional[torch.Tensor] = None,
    ):
        # [B, T, num_post_punct * max_token_len]
        punct_logits_post = self._punct_head_post(hidden_states=encoded)
        # [B, T, num_pre_punct]
        punct_logits_pre = self._punct_head_pre(hidden_states=encoded)
        # [B, T, max_subword_len]
        cap_logits = self._cap_head(hidden_states=encoded)
        # [B, T, 2]
        seg_logits = self._seg_head(hidden_states=encoded)

        # Unfold the logits to align with chars: [B, T, max_subword_len * C] -> [B, T, max_subword_len, C]
        punct_logits_post = punct_logits_post.view([*punct_logits_post.shape[:-1], -1, self._num_post_punct_classes])

        return punct_logits_pre, punct_logits_post, cap_logits, seg_logits


class MHAPunctCapSegDecoder(PunctCapSegDecoder):
    """
    Decoder with multi-headed attention to condition the true-casing and sentence-boundary heads with the punctuation
    predictions.
    """

    def __init__(
        self,
        encoder_dim: int,
        punct_num_classes_post: int,
        punct_num_classes_pre: int,
        max_subword_length: int,
        emb_dim: int = 4,
        punct_head_n_layers: int = 1,
        punct_head_dropout: float = 0.1,
        cap_head_n_layers: int = 1,
        cap_head_dropout: float = 0.1,
        seg_head_n_layers: int = 1,
        seg_head_dropout: float = 0.1,
        transformer_num_layers: int = 2,
        transformer_inner_size: int = 2048,
        transformer_num_heads: int = 4,
        transformer_ffn_dropout: float = 0.1,
        post_punct_null_idx: int = 0,
    ):
        super().__init__()
        self._max_subword_len = max_subword_length
        self._num_post_punct_classes = punct_num_classes_post
        self._post_punct_null_idx = post_punct_null_idx
        # Use an extra embedding for padding
        self._emb_pad_idx = punct_num_classes_post
        self._punct_emb = nn.Embedding(
            num_embeddings=punct_num_classes_post + 1, embedding_dim=emb_dim, padding_idx=self._emb_pad_idx
        )
        # Initialize the embeddings to the same scale as the attn module and heads
        nn.init.normal_(self._punct_emb.weight, mean=0.0, std=0.02)

        # Will append embeddings for each of the N characters in a subword, up to max subtoken size
        self._encoder_hidden_dim = encoder_dim + emb_dim * max_subword_length
        self._punct_head_post: TokenClassifier = TokenClassifier(
            hidden_size=encoder_dim,
            num_layers=punct_head_n_layers,
            dropout=punct_head_dropout,
            activation="relu",
            log_softmax=False,
            num_classes=punct_num_classes_post * max_subword_length,
        )
        self._punct_head_pre: TokenClassifier = TokenClassifier(
            hidden_size=encoder_dim,
            num_layers=punct_head_n_layers,
            dropout=punct_head_dropout,
            activation="relu",
            log_softmax=False,
            num_classes=punct_num_classes_pre,
        )
        self._cap_seg_encoder: TransformerEncoder = TransformerEncoder(
            num_layers=transformer_num_layers,
            hidden_size=self._encoder_hidden_dim,
            inner_size=transformer_inner_size,
            ffn_dropout=transformer_ffn_dropout,
            num_attention_heads=transformer_num_heads,
        )
        self._seg_head: TokenClassifier = TokenClassifier(
            hidden_size=self._encoder_hidden_dim,
            num_layers=seg_head_n_layers,
            dropout=seg_head_dropout,
            activation="relu",
            log_softmax=False,
            num_classes=2,
        )
        # Will append sentence boundary predictions
        self._cap_head: TokenClassifier = TokenClassifier(
            hidden_size=self._encoder_hidden_dim + 1,
            num_layers=cap_head_n_layers,
            dropout=cap_head_dropout,
            activation="relu",
            log_softmax=False,
            num_classes=max_subword_length,
        )

    @typecheck()
    def forward(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        punc_mask: torch.Tensor,
        punc_targets: Optional[torch.Tensor] = None,
        seg_targets: Optional[torch.Tensor] = None,
    ):
        # [B, T, C * max_token_len]
        punct_logits_post = self._punct_head_post(hidden_states=encoded)
        # Unfold the logits to match the targets: [B, T, max_subword_len * C] -> [B, T, max_subword_len, C]
        punct_logits_post = punct_logits_post.view([*punct_logits_post.shape[:-1], -1, self._num_post_punct_classes])
        # [B, T, C]
        punct_logits_pre = self._punct_head_pre(hidden_states=encoded)

        # At training time, we get the reference punctuation targets to teacher-force the other heads.
        # At inference time, use the model's predictions.
        if punc_targets is None:
            # [B, T, max_subword_len, C] -> [B, T, max_subword_len]
            # Note - no need to detach because this should not happen in train mode.
            punc_targets = punct_logits_post.argmax(dim=-1)
        # Fill with null indices which are padded (model is allowed to predict garbage)
        punc_targets = punc_targets.masked_fill(mask=~punc_mask.bool(), value=self._emb_pad_idx)
        # [B, T, max_subword_len, emb_dim]
        embs = self._punct_emb(punc_targets)
        # Stack and cat the embeddings
        # [B, T, max_subword_len, emb_dim] -> [B, T, max_subword_len * emb_dim]
        embs = embs.flatten(2)
        # [B, T, D + max_subword_len * emb_dim]
        cap_seg_encoder_input = torch.cat((encoded, embs), dim=-1)

        cap_seg_encoded = self._cap_seg_encoder(encoder_states=cap_seg_encoder_input, encoder_mask=mask)
        # [B, T, 2]
        seg_logits = self._seg_head(hidden_states=cap_seg_encoded)

        # In inference mode, generate the sentence boundary predictions
        if seg_targets is None:
            seg_targets = seg_logits.argmax(dim=-1)
        # For consistency, set the first token position to 1 (effectively a sentence boundary)
        seg_targets[:, 0] = 1
        # Shift the targets right by one, to indicate which tokens are the beginning of a sentence rather than the end.
        seg_targets = torch.nn.functional.pad(seg_targets, pad=[1, 0])
        # Trim the right because we padded left
        seg_targets = seg_targets[:, :-1]
        # Note that the seg targets contain -100 (ignore_idx) in BOS/EOS positions, but BOS is overwritten and EOS
        # is shifted right, into the padding region.
        # [B, T] -> [B, T, 1]
        seg_targets = seg_targets.unsqueeze(-1)
        # Concatenate the shifted sentence boundary predictions for the truecase head
        cap_head_input = torch.cat((cap_seg_encoded, seg_targets), dim=-1)
        # [B, T, max_subword_len]
        cap_logits = self._cap_head(hidden_states=cap_head_input)

        return punct_logits_pre, punct_logits_post, cap_logits, seg_logits
