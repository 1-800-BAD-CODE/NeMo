import abc
from typing import Optional, Dict

import torch
import torch.nn as nn

from nemo.collections.nlp.modules.common.transformer import TransformerEncoder
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.core import typecheck
from nemo.core.classes import NeuralModule
from nemo.core.neural_types import NeuralType, ChannelType, LogitsType, MaskType
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector


class PunctCapSegDecoder(NeuralModule):
    def __init__(self,):
        super().__init__()

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        #  bert model output {"last_hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}
        return {"encoded": NeuralType(("B", "T", "D"), ChannelType()), "mask": NeuralType(("B", "T"), MaskType())}

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "punct_pre_logits": NeuralType(("B", "T", "D", "C"), LogitsType()),  # D == max_subword_length
            "punct_post_logits": NeuralType(("B", "T", "D", "C"), LogitsType()),  # D == max_subword_length
            "cap_logits": NeuralType(("B", "T", "D"), LogitsType()),  # D == max_subword_length
            "seg_logits": NeuralType(("B", "T", "C"), LogitsType()),  # C == 2
        }

    @abc.abstractmethod
    def forward(self, encoded: torch.Tensor, lengths: torch.Tensor):
        raise NotImplementedError()

    @classmethod
    def restore_from(
        cls,
        restore_path: str,
        override_config_path: Optional[str] = None,
        map_location: Optional["torch.device"] = None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Optional["Trainer"] = None,
        save_restore_connector: SaveRestoreConnector = None,
    ):
        pass

    def save_to(self, save_path: str):
        pass


class MHAPunctCapSegDecoder(PunctCapSegDecoder):
    """
    

    """

    def __init__(
        self,
        encoder_dim: int,
        punct_num_classes_post: int,
        punct_num_classes_pre: int,
        max_subword_length: int,
        punct_emb_dim: int = 4,
        punct_head_n_layers: int = 1,
        punct_head_dropout: float = 0.1,
        cap_head_n_layers: int = 1,
        cap_head_dropout: float = 0.1,
        seg_head_n_layers: int = 1,
        seg_head_dropout: float = 0.1,
        transformer_num_layers: int = 2,
        transformer_inner_size: int = 512,
        transformer_num_heads: int = 4,
        transformer_ffn_dropout: float = 0.1,
    ):
        super().__init__()
        self._max_subword_len = max_subword_length
        self._num_post_punct_classes = punct_num_classes_post
        self._num_pre_punct_classes = punct_num_classes_pre
        self._punct_pre_projection: nn.Linear = nn.Linear(max_subword_length * punct_num_classes_pre, punct_emb_dim)
        self._punct_post_projection: nn.Linear = nn.Linear(max_subword_length * punct_num_classes_post, punct_emb_dim)
        # Will append whether each character is followed by a full stop candidate
        self._encoder_hidden_dim = encoder_dim + 2 * punct_emb_dim
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
            num_classes=punct_num_classes_pre * max_subword_length,
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
        self._cap_head: TokenClassifier = TokenClassifier(
            hidden_size=self._encoder_hidden_dim,
            num_layers=cap_head_n_layers,
            dropout=cap_head_dropout,
            activation="relu",
            log_softmax=False,
            num_classes=max_subword_length,
        )

    @typecheck()
    def forward(self, encoded: torch.Tensor, mask: torch.Tensor):
        # [B, T, C * max_token_len]
        punct_logits_post = self._punct_head_post(hidden_states=encoded)
        punct_logits_pre = self._punct_head_pre(hidden_states=encoded)
        # [B, T, emb_dim]
        punct_emb_post = self._punct_post_projection(punct_logits_post.clone().detach())
        punct_emb_pre = self._punct_pre_projection(punct_logits_pre.clone().detach())
        # [B, T, D + 2*emb_dim]
        cap_seg_input = torch.cat((encoded, punct_emb_post, punct_emb_pre), dim=2)
        cap_seg_encoded = self._cap_seg_encoder(encoder_states=cap_seg_input, encoder_mask=mask)
        # [B, T, max_subword_len]
        cap_logits = self._cap_head(hidden_states=cap_seg_encoded)
        # [B, T, 2]
        seg_logits = self._seg_head(hidden_states=cap_seg_encoded)

        # Unfold the logits to match the targets: [B, T, max_subword_len * C] -> [B, T, max_subword_len, C]
        punct_logits_pre = punct_logits_pre.view([*punct_logits_pre.shape[:-1], -1, self._num_pre_punct_classes])
        punct_logits_post = punct_logits_post.view([*punct_logits_post.shape[:-1], -1, self._num_post_punct_classes])

        return punct_logits_pre, punct_logits_post, cap_logits, seg_logits
