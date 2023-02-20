import abc
from typing import Optional, Dict

import torch
import torch.nn as nn

from nemo.collections.nlp.modules.common.transformer import TransformerEncoder
from nemo.core import typecheck
from nemo.core.classes import NeuralModule
from nemo.core.neural_types import NeuralType, ChannelType, LogitsType, MaskType, LabelsType
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector


class ClassificationHead(NeuralModule):
    def __init__(
        self,
        num_classes: int,
        encoded_dim: int,
        num_layers: int = 1,
        intermediate_dim: int = None,
        activation: str = "relu",
        dropout: Optional[float] = None,
    ):
        super().__init__()
        if num_layers > 1 and intermediate_dim is None:
            raise ValueError("Need intermediate dim if using > 1 layer")
        if activation == "relu":
            self._act = nn.ReLU()
        elif activation == "gelu":
            self._act = nn.GELU()
        else:
            raise NotImplementedError(f"Unsupported activation type: '{activation}'")
        self._dropout: Optional[nn.Dropout] = None if dropout is None or dropout <= 0 else nn.Dropout(p=dropout)
        self._linears: nn.ModuleList = nn.ModuleList()
        for i in range(num_layers):
            in_dim = encoded_dim if i == 0 else intermediate_dim
            out_dim = num_classes if i == num_layers - 1 else intermediate_dim
            next_linear: nn.Linear = nn.Linear(in_dim, out_dim)
            nn.init.normal_(next_linear.weight, mean=0.0, std=0.02)
            self._linears.append(next_linear)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "encoded": NeuralType(("B", "T", "D"), ChannelType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"logits": NeuralType(("B", "T", "C"), LogitsType())}

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        x = encoded
        if self._dropout is not None:
            x = self._dropout(x)
        for i, layer in enumerate(self._linears):
            x = layer(x)
            # If not the last layer, apply dropout and activation
            if i < len(self._linears) - 1:
                if self._dropout is not None:
                    x = self._dropout(x)
                x = self._act(x)
        return x

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


class PunctCapSegDecoder(NeuralModule):
    def __init__(self,):
        super().__init__()

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        When training, the reference punctuation targets are used to condition the other classifiers. This is needed to
        ensure the targets align with the input to prevent the heads from ignoring the (sometimes incorrect)
        punctuation predictions.

        """
        return {
            "encoded": NeuralType(("B", "T", "D"), ChannelType()),
            "mask": NeuralType(("B", "T"), MaskType()),
            "punc_targets": NeuralType(("B", "T"), LabelsType(), optional=True),  # training - reference targets
            "seg_targets": NeuralType(("B", "T"), LabelsType(), optional=True),  # training - reference targets
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "punct_pre_logits": NeuralType(("B", "T", "C"), LogitsType()),
            "punct_post_logits": NeuralType(("B", "T", "C"), LogitsType()),
            "cap_logits": NeuralType(("B", "T", "D"), LogitsType()),  # D == max_subword_length
            "seg_logits": NeuralType(("B", "T", "C"), LogitsType()),  # C == 2
        }

    @abc.abstractmethod
    def forward(
        self, encoded: torch.Tensor, mask: torch.Tensor, punc_targets: Optional[torch.Tensor] = None,
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
    """Simple decoder with unconditional classifiers.


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
        punct_head_intermediate_dim: int = 256,
        cap_head_intermediate_dim: int = 256,
        seg_head_intermediate_dim: int = 256,
    ):
        super().__init__()
        self._max_subword_len = max_subword_length
        self._num_post_punct_classes = punct_num_classes_post
        self._num_pre_punct_classes = punct_num_classes_pre
        self._punct_head_post: ClassificationHead = ClassificationHead(
            encoded_dim=encoder_dim,
            num_layers=punct_head_n_layers,
            dropout=punct_head_dropout,
            activation="relu",
            intermediate_dim=punct_head_intermediate_dim,
            num_classes=punct_num_classes_post,
        )
        self._punct_head_pre: ClassificationHead = ClassificationHead(
            encoded_dim=encoder_dim,
            num_layers=punct_head_n_layers,
            dropout=punct_head_dropout,
            activation="relu",
            intermediate_dim=punct_head_intermediate_dim,
            num_classes=punct_num_classes_pre,
        )
        self._seg_head: ClassificationHead = ClassificationHead(
            encoded_dim=encoder_dim,
            num_layers=seg_head_n_layers,
            dropout=seg_head_dropout,
            activation="relu",
            intermediate_dim=seg_head_intermediate_dim,
            num_classes=2,
        )
        self._cap_head: ClassificationHead = ClassificationHead(
            encoded_dim=encoder_dim,
            num_layers=cap_head_n_layers,
            dropout=cap_head_dropout,
            activation="relu",
            intermediate_dim=cap_head_intermediate_dim,
            num_classes=max_subword_length,
        )

    @typecheck()
    def forward(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        punc_targets: Optional[torch.Tensor] = None,
        seg_targets: Optional[torch.Tensor] = None,
    ):
        # [B, T, num_post_punct]
        punct_logits_post = self._punct_head_post(encoded=encoded)
        # [B, T, num_pre_punct]
        punct_logits_pre = self._punct_head_pre(encoded=encoded)
        # [B, T, max_subword_len]
        cap_logits = self._cap_head(encoded=encoded)
        # [B, T, 2]
        seg_logits = self._seg_head(encoded=encoded)

        return punct_logits_pre, punct_logits_post, cap_logits, seg_logits


class MHAPunctCapSegDecoder(PunctCapSegDecoder):
    """Decoder with multi-headed attention to utilize punctuation to condition the other classifiers.
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
        punct_head_intermediate_dim: int = 256,
        cap_head_intermediate_dim: int = 256,
        seg_head_intermediate_dim: int = 256,
    ):
        super().__init__()
        self._max_subword_len = max_subword_length
        self._num_post_punct_classes = punct_num_classes_post
        self._post_punct_null_idx = post_punct_null_idx
        self._punct_emb = nn.Embedding(num_embeddings=punct_num_classes_post, embedding_dim=emb_dim)
        # Initialize the embeddings to the same scale as the attn module and heads
        nn.init.normal_(self._punct_emb.weight, mean=0.0, std=0.02)

        # Will append embeddings for each of the N characters in a subword, up to max subtoken size
        # Note that the number of heads must be a divisor of this value. For encoder 512, emb 4, 4 heads is ok.
        self._encoder_hidden_dim = encoder_dim + emb_dim
        self._punct_head_post: ClassificationHead = ClassificationHead(
            encoded_dim=encoder_dim,
            num_layers=punct_head_n_layers,
            dropout=punct_head_dropout,
            activation="relu",
            intermediate_dim=punct_head_intermediate_dim,
            num_classes=punct_num_classes_post,
        )
        self._punct_head_pre: ClassificationHead = ClassificationHead(
            encoded_dim=self._encoder_hidden_dim,
            num_layers=punct_head_n_layers,
            dropout=punct_head_dropout,
            activation="relu",
            intermediate_dim=punct_head_intermediate_dim,
            num_classes=punct_num_classes_pre,
        )
        self._cap_seg_encoder: TransformerEncoder = TransformerEncoder(
            num_layers=transformer_num_layers,
            hidden_size=self._encoder_hidden_dim,
            inner_size=transformer_inner_size,
            ffn_dropout=transformer_ffn_dropout,
            num_attention_heads=transformer_num_heads,
        )
        self._seg_head: ClassificationHead = ClassificationHead(
            encoded_dim=self._encoder_hidden_dim,
            num_layers=seg_head_n_layers,
            dropout=seg_head_dropout,
            activation="relu",
            intermediate_dim=seg_head_intermediate_dim,
            num_classes=2,
        )
        # Will append sentence boundary predictions
        self._cap_head: ClassificationHead = ClassificationHead(
            encoded_dim=self._encoder_hidden_dim + 1,
            num_layers=cap_head_n_layers,
            dropout=cap_head_dropout,
            activation="relu",
            intermediate_dim=cap_head_intermediate_dim,
            num_classes=max_subword_length,
        )

    @typecheck()
    def forward(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        punc_targets: Optional[torch.Tensor] = None,
        seg_targets: Optional[torch.Tensor] = None,
    ):
        # [B, T, C * max_token_len]
        punct_logits_post = self._punct_head_post(encoded=encoded)

        # At training time, we get the reference punctuation targets to teacher-force the other heads.
        # At inference time, use the model's predictions.
        if punc_targets is None:
            # [B, T, C] -> [B, T]
            # Note - no need to detach because this should not happen in train mode.
            punc_targets = punct_logits_post.argmax(dim=-1)
        # [B, T, emb_dim]
        embs = self._punct_emb(punc_targets)
        # Concatenate the punctuation embeddings to the input and re-encode
        # [B, T, D + emb_dim]
        encoded_with_embs = torch.cat((encoded, embs), dim=-1)
        re_encoded = self._cap_seg_encoder(encoder_states=encoded_with_embs, encoder_mask=mask)
        # [B, T, C]
        punct_logits_pre = self._punct_head_pre(encoded=re_encoded)
        # [B, T, 2]
        seg_logits = self._seg_head(encoded=re_encoded)

        # In inference mode, generate the sentence boundary predictions
        if seg_targets is None:
            seg_targets = seg_logits.argmax(dim=-1)
        # For consistency, set the first token slot (BOS) to 1 (effectively a sentence boundary)
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
        cap_head_input = torch.cat((re_encoded, seg_targets), dim=-1)
        # [B, T, max_subword_len]
        cap_logits = self._cap_head(encoded=cap_head_input)

        return punct_logits_pre, punct_logits_post, cap_logits, seg_logits
