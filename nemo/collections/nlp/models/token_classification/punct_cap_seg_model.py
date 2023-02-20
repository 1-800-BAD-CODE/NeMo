from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer

from nemo.collections.common.data import ConcatMapDataset
from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.common.tokenizers import SentencePieceTokenizer
from nemo.collections.nlp.data.token_classification.punct_cap_seg_dataset import (
    InferencePunctCapSegDataset,
    PunctCapSegDataset,
    NULL_PUNCT_TOKEN,
)
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.token_classification import PunctCapSegDecoder
from nemo.core import PretrainedModelInfo, typecheck
from nemo.core.neural_types import ChannelType, LengthsType, MaskType, NeuralType, LogitsType
from nemo.utils import logging


class PunctCapSegModel(NLPModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None) -> None:
        super().__init__(cfg=cfg, trainer=trainer)
        # During training, print metrics for these languages only
        self._log_val_metrics_for = set(cfg.get("log_val_metrics_for", []))
        # Whether to print the metrics report for training data. Generates a lot of output.
        self._log_train_metrics = cfg.get("log_train_metrics", False)
        # Should be set to the training DS max length; default to the positional embeddings size
        self._max_length = self._cfg.get("max_length", self.bert_model.config.max_position_embeddings)

        # Retrieve labels
        self._punct_post_labels: List[str] = self._cfg.punct_post_labels
        self._punct_pre_labels: List[str] = self._cfg.punct_pre_labels
        # Map each label to its integer index
        self._punct_pre_token_to_index: Dict[str, int] = {token: i for i, token in enumerate(self._punct_pre_labels)}
        self._punct_post_token_to_index: Dict[str, int] = {token: i for i, token in enumerate(self._punct_post_labels)}
        # Resolve index of null token
        self._null_punct_pre_index: int = self._punct_pre_token_to_index[NULL_PUNCT_TOKEN]
        self._null_punct_post_index: int = self._punct_post_token_to_index[NULL_PUNCT_TOKEN]

        # Used for loss masking. Should by synchronized with data sets.
        self._ignore_idx: int = self._cfg.get("ignore_idx", -100)

        # Used for making character-level predictions with subwords (predict max_token_len per token)
        self._using_sp = isinstance(self.tokenizer, SentencePieceTokenizer)
        if not self._using_sp:
            self._max_token_len = max(len(x) for x in self.tokenizer.vocab)
        else:
            # SentencePiece model - AutoTokenizer doesn't have 'vocab' attr for some SP models
            vocab_size = self.tokenizer.vocab_size
            self._max_token_len = max(len(self.tokenizer.ids_to_tokens([idx])[0]) for idx in range(vocab_size))

        # [B, T, num_pre_punct]
        self._punct_pre_loss: CrossEntropyLoss = CrossEntropyLoss(
            weight=cfg.loss.punct_pre.get("weight"), ignore_index=self._ignore_idx, logits_ndim=3
        )
        # [B, T, num_post_punct, max_chars_per_subword]
        self._punct_post_loss: CrossEntropyLoss = CrossEntropyLoss(
            weight=cfg.loss.punct_post.get("weight"), ignore_index=self._ignore_idx, logits_ndim=3
        )
        # [B, T, 2] binary preds
        self._seg_loss: CrossEntropyLoss = CrossEntropyLoss(
            weight=cfg.loss.seg.get("weight"), ignore_index=self._ignore_idx, logits_ndim=3
        )
        # [B, T, max_chars_per_subword]
        # For true-casing, we use multi-label classification to predict for each char in a subword
        self._cap_loss: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(cfg.loss.cap["weight"]) if "weight" in cfg.loss.cap else None
        )
        # Weights can be specified in punct{-pre,-post}, cap, seg order.
        self._agg_loss = AggregatorLoss(num_inputs=4, weights=cfg.loss.get("agg_loss_weights"))

        with open_dict(self._cfg.decoder):
            self._cfg.decoder.punct_num_classes_post = len(self._punct_post_labels)
            self._cfg.decoder.punct_num_classes_pre = len(self._punct_pre_labels)
            self._cfg.decoder.max_subword_length = self._max_token_len
        self._decoder: PunctCapSegDecoder = PunctCapSegModel.from_config_dict(self._cfg.decoder)

        # Set each dataset's tokenizer. Model's tokenizer doesn't exist until we initialize BertModule, but datasets are
        # instantiated prior to that.
        if self._train_dl is not None:
            # Train DL has one ConcatDataset
            for dataset in self._train_dl.dataset.datasets:
                dataset.tokenizer = self.tokenizer
        if self._validation_dl is not None:
            # Validation DL is a list of PunctCapSegDataset
            for dataset in self._validation_dl:
                dataset.tokenizer = self.tokenizer

        # Will be populated when dev/test sets are setup
        self._dev_metrics: nn.ModuleList = nn.ModuleList()
        if self._validation_dl is not None:
            self._dev_metrics = self._setup_metrics(len(self._validation_dl))
        self._train_metrics: nn.ModuleDict = self._setup_metrics()
        # module list of module dict
        self._test_metrics: nn.ModuleDict = self._setup_metrics()

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self.setup_validation_data(val_data_config)

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        self._test_dl = self._setup_test_dataloader_from_config(cfg=test_data_config)

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self._validation_dl = self._setup_eval_dataloaders_from_config(cfg=val_data_config)
        self._validation_names = [f"val_{dl.dataset.language}" for dl in self._validation_dl]
        # TODO if self._dev_metrics already exists, overwrite it?
        # self._setup_metrics(len(self._validation_dl), self._dev_metrics)

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        self._train_dl = self._setup_train_dataloader_from_config(cfg=train_data_config)

    def _setup_metrics(self, num_dl: int = 1) -> Union[nn.ModuleDict, nn.ModuleList]:
        """Creates metrics for each data loader. Typically, we have one DL per language.

        Metrics are reported for punctuation (pre- and post-token), true casing, segmentation, and loss.

        Returns:
            A :class:``nn.ModuleList``, with one element per data loader. Each element is another
            :class:``nn.ModuleList`` of metrics for that language. If `num_dl == 1`, then a single `nn.ModuleDict` is
            returned.

        """
        module_list: nn.ModuleList = nn.ModuleList()
        for _ in range(num_dl):
            metrics: nn.ModuleDict = nn.ModuleDict(
                {
                    "loss": GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True),
                    "punct_pre_report": ClassificationReport(
                        num_classes=len(self._punct_pre_labels),
                        label_ids=self._punct_pre_token_to_index,
                        mode="macro",
                        dist_sync_on_step=False,
                    ),
                    "punct_post_report": ClassificationReport(
                        num_classes=len(self._punct_post_labels),
                        label_ids=self._punct_post_token_to_index,
                        mode="macro",
                        dist_sync_on_step=False,
                    ),
                    "cap_report": ClassificationReport(
                        num_classes=2, label_ids={"LOWER": 0, "UPPER": 1}, mode="macro", dist_sync_on_step=False
                    ),
                    "seg_report": ClassificationReport(
                        num_classes=2, label_ids={"NOSTOP": 0, "FULLSTOP": 1}, mode="macro", dist_sync_on_step=False
                    ),
                }
            )
            module_list.append(metrics)
        if num_dl == 1:
            return module_list[0]
        return module_list

    def _setup_eval_dataloaders_from_config(self, cfg) -> List[torch.utils.data.DataLoader]:
        dataloaders: List[torch.utils.data.DataLoader] = []
        for ds_config in cfg.datasets:
            # Add all common variables, if not set already
            with open_dict(ds_config):
                for k, v in cfg.get("common", {}).items():
                    if k not in ds_config:
                        ds_config[k] = v
            dataset: PunctCapSegDataset = instantiate(ds_config)
            if not isinstance(dataset, PunctCapSegDataset):
                raise ValueError(
                    f"Expected dataset config to instantiate an implementation of 'PunctCapSegDataset' but instead got "
                    f"'{type(dataset)}' from config {ds_config}."
                )
            if hasattr(self, "tokenizer"):
                dataset.tokenizer = self.tokenizer
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                collate_fn=dataset.collate_fn,
                batch_size=cfg.get("batch_size", 128),
                num_workers=cfg.get("num_workers", 8),
                pin_memory=cfg.get("pin_memory", False),
                drop_last=cfg.get("drop_last", False),
                worker_init_fn=dataset.worker_init_fn,
            )
            dataloaders.append(dataloader)
        return dataloaders

    def _setup_test_dataloader_from_config(self, cfg) -> List[torch.utils.data.DataLoader]:
        # Add the model-specific parameters needed for generating targets for metrics
        with open_dict(cfg):
            cfg.dataset.punct_pre_labels = self._cfg.punct_pre_labels
            cfg.dataset.punct_post_labels = self._cfg.punct_post_labels
            cfg.dataset.max_length = self._max_length
        # _target_ should instantiate a PunctCapSegDataset
        dataset: PunctCapSegDataset = instantiate(cfg.dataset)
        if not isinstance(dataset, PunctCapSegDataset):
            raise ValueError(
                f"Expected dataset config to instantiate an implementation of 'PunctCapSegDataset' but instead got "
                f"'{type(dataset)}' from config {cfg.dataset}."
            )
        # Pass tokenizer to test set
        dataset.tokenizer = self.tokenizer
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=cfg.get("batch_size", 128),
            num_workers=cfg.get("num_workers", 8),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
            worker_init_fn=dataset.worker_init_fn,
        )
        return dataloader

    def _setup_train_dataloader_from_config(self, cfg) -> List[torch.utils.data.DataLoader]:
        datasets: List[PunctCapSegDataset] = []
        for ds_config in cfg.datasets:
            # Add all common variables, if not set already
            with open_dict(ds_config):
                for k, v in cfg.get("common", {}).items():
                    if k not in ds_config:
                        ds_config[k] = v
            dataset: PunctCapSegDataset = instantiate(ds_config)
            if not isinstance(dataset, PunctCapSegDataset):
                raise ValueError(
                    f"Expected dataset config to instantiate an implementation of 'PunctCapSegDataset' but instead got "
                    f"'{type(dataset)}' from config {ds_config}."
                )
            # If model tokenizer has been set already, assign it
            if hasattr(self, "tokenizer"):
                dataset.tokenizer = self.tokenizer
            datasets.append(dataset)
        # Currently only one type of dataset is implemented; ok to always use a map data set
        dataset: ConcatMapDataset = ConcatMapDataset(
            datasets=datasets,
            sampling_technique=cfg.get("sampling_technique", "temperature"),
            sampling_temperature=cfg.get("sampling_temperature", 5),
            sampling_probabilities=cfg.get("sampling_probabilities", None),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=datasets[0].collate_fn,  # TODO assumption; works for now
            batch_size=cfg.get("batch_size", 256),
            num_workers=cfg.get("num_workers", 8),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
            worker_init_fn=dataset.worker_init_fn,
        )
        return dataloader

    def _run_step(self, batch: Tuple, testing: bool = False):
        # All inputs and targets are shape [B, T]
        inputs, punct_pre_targets, punct_post_targets, cap_targets, seg_targets, _ = batch
        mask = inputs.ne(self.tokenizer.pad_id)
        # Encoded output is [B, T, D]
        encoded = self.bert_model(input_ids=inputs, attention_mask=mask, token_type_ids=None)
        # Some LMs will return a tuple
        if isinstance(encoded, tuple):
            encoded = encoded[0]

        # Make a binary mask from the post punc targets
        punc_mask = punct_post_targets.ne(self._cfg.get("ignore_idx", -100))
        punct_targets_for_decoder = punct_post_targets.masked_fill(~punc_mask, self._null_punct_post_index)
        # In training mode, always feed ground truth predictions to the heads.
        # For val, it's useful to use reference to not penalize the cap/seg heads for bad punctuation predictions when
        # selecting the best model.
        punct_logits_pre, punct_logits_post, cap_logits, seg_logits = self._decoder(
            encoded=encoded,
            mask=mask,
            punc_targets=None if testing else punct_targets_for_decoder,
            seg_targets=None if testing else seg_targets,
        )

        # Compute losses
        punct_pre_loss = self._punct_pre_loss(logits=punct_logits_pre, labels=punct_pre_targets)
        punct_post_loss = self._punct_post_loss(logits=punct_logits_post, labels=punct_post_targets)
        seg_loss = self._seg_loss(logits=seg_logits, labels=seg_targets)
        # If all elements are uncased, BCE returns nan. So set to zero if no targets (ja, zh, hi, etc.).
        cap_mask = cap_targets.ne(self._ignore_idx)
        if cap_mask.any():
            cap_loss = self._cap_loss(input=cap_logits[cap_mask], target=cap_targets[cap_mask].float())
        else:
            # Dimensionless 0.0 like cap_logits
            cap_loss = cap_logits.new_zeros(1).squeeze()

        loss = self._agg_loss.forward(loss_1=punct_pre_loss, loss_2=punct_post_loss, loss_3=cap_loss, loss_4=seg_loss)

        return loss, punct_logits_pre, punct_logits_post, cap_logits, seg_logits

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int):
        loss, punct_pre_logits, punct_post_logits, cap_logits, seg_logits = self._run_step(batch)
        lr = self._optimizer.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)
        self.log("train_loss", loss)

        # Maybe accumulate batch metrics
        if batch_idx % self._cfg.get("batch_metrics_every_n_steps", 10) == 0:
            _, punct_pre_targets, punct_post_targets, cap_targets, seg_targets, _ = batch
            punct_pre_preds = punct_pre_logits.argmax(dim=-1)
            punct_post_preds = punct_post_logits.argmax(dim=-1)
            cap_mask = cap_targets.ne(self._ignore_idx)
            cap_preds = cap_logits[cap_mask].sigmoid().gt(0.5)
            seg_mask = seg_targets.ne(self._ignore_idx)
            seg_preds = seg_logits.argmax(dim=-1)

            punct_pre_mask = punct_pre_targets.ne(self._ignore_idx)
            punct_post_mask = punct_post_targets.ne(self._ignore_idx)
            self._train_metrics["punct_pre_report"](punct_pre_preds[punct_pre_mask], punct_pre_targets[punct_pre_mask])
            self._train_metrics["punct_post_report"](
                punct_post_preds[punct_post_mask], punct_post_targets[punct_post_mask]
            )
            self._train_metrics["cap_report"](cap_preds, cap_targets[cap_mask])
            self._train_metrics["seg_report"](seg_preds[seg_mask], seg_targets[seg_mask])

        # Maybe log and reset batch metrics
        if batch_idx > 0 and batch_idx % self._trainer.log_every_n_steps == 0:
            for analytic in ["punct_pre", "punct_post", "cap", "seg"]:
                precision, recall, f1, report = self._train_metrics[f"{analytic}_report"].compute()
                self._train_metrics[f"{analytic}_report"].reset()
                # For some analytics, NaN/inf are natural; e.g. uncased languages with no true-case targets.
                if precision.isnan():
                    precision = torch.zeros_like(precision)
                if recall.isnan():
                    recall = torch.zeros_like(recall)
                if f1.isnan():
                    f1 = torch.zeros_like(f1)
                self.log(f"train_batch/{analytic}_precision", precision)
                self.log(f"train_batch/{analytic}_recall", recall)
                self.log(f"train_batch/{analytic}_f1", f1)
                if self._log_train_metrics:
                    logging.info(f"Train batch {analytic} report: {report}")

        return loss

    def _eval_step(self, batch: Tuple, dataloader_idx: int = 0) -> None:
        loss, punct_pre_logits, punct_post_logits, cap_logits, seg_logits = self._run_step(batch)
        _, punct_pre_targets, punct_post_targets, cap_targets, seg_targets, _ = batch
        # All log probs are [B, T, D]
        punct_pre_preds = punct_pre_logits.argmax(dim=-1)
        punct_post_preds = punct_post_logits.argmax(dim=-1)
        cap_mask = cap_targets.ne(self._ignore_idx)
        cap_preds = cap_logits[cap_mask].sigmoid().gt(0.5)
        seg_mask = seg_targets.ne(self._ignore_idx)
        seg_preds = seg_logits.argmax(dim=-1)

        metrics: nn.ModuleDict = self._dev_metrics[dataloader_idx]
        punct_pre_mask = punct_pre_targets.ne(self._ignore_idx)
        punct_post_mask = punct_post_targets.ne(self._ignore_idx)
        num_targets = punct_pre_mask.sum() + punct_post_mask.sum() + cap_mask.sum() + seg_mask.sum()
        metrics["loss"](loss=loss, num_measurements=num_targets)
        metrics["punct_pre_report"](punct_pre_preds[punct_pre_mask], punct_pre_targets[punct_pre_mask])
        metrics["punct_post_report"](punct_post_preds[punct_post_mask], punct_post_targets[punct_post_mask])
        metrics["cap_report"](cap_preds, cap_targets[cap_mask])
        metrics["seg_report"](seg_preds[seg_mask], seg_targets[seg_mask])

    def _test_step(self, batch: Tuple, dataloader_idx: int = 0) -> None:
        loss, punct_pre_logits, punct_post_logits, cap_logits, seg_logits = self._run_step(batch, testing=True)
        _, punct_pre_targets, punct_post_targets, cap_targets, seg_targets, _ = batch
        # Prepare masks
        cap_mask = cap_targets.ne(self._ignore_idx)
        seg_mask = seg_targets.ne(self._ignore_idx)
        punct_pre_mask = punct_pre_targets.ne(self._ignore_idx)
        punct_post_mask = punct_post_targets.ne(self._ignore_idx)
        # Get all probs. All log probs are [B, T, D]
        punct_pre_probs = punct_pre_logits.softmax(dim=-1)
        punct_post_probs = punct_post_logits.softmax(dim=-1)
        seg_probs = seg_logits.softmax(dim=-1)[..., 1]
        cap_probs = cap_logits[cap_mask].sigmoid()

        punct_pre_preds = punct_pre_probs.argmax(dim=-1)
        punct_post_preds = punct_post_probs.argmax(dim=-1)
        seg_preds = seg_probs.ge(0.5)
        cap_preds = cap_probs.ge(0.5)
        self._test_metrics["punct_pre_report"](punct_pre_preds[punct_pre_mask], punct_pre_targets[punct_pre_mask])
        self._test_metrics["punct_post_report"](punct_post_preds[punct_post_mask], punct_post_targets[punct_post_mask])
        self._test_metrics["cap_report"](cap_preds, cap_targets[cap_mask])
        self._test_metrics["seg_report"](seg_preds[seg_mask], seg_targets[seg_mask])

    def _get_language_for_dl_idx(self, idx: int) -> str:
        ds: PunctCapSegDataset = self._validation_dl[idx].dataset
        language = ds.language
        return language

    def _multi_eval_epoch_end(self, dataloader_idx: int):
        """ Epoch end logic for both validation and test """
        metric_dict: nn.ModuleDict = self._dev_metrics[dataloader_idx]
        # Resolve language for better logging
        language = self._get_language_for_dl_idx(dataloader_idx)

        # Compute, reset, and log the loss for this language
        loss = metric_dict["loss"].compute()
        metric_dict["loss"].reset()
        self.log(f"val_{language}/loss", loss)

        # Compute, reset, and log the precision/recall/f1 for punct/cap/seg for this language
        for analytic in ["punct_pre", "punct_post", "cap", "seg"]:
            precision, recall, f1, report = metric_dict[f"{analytic}_report"].compute()
            metric_dict[f"{analytic}_report"].reset()
            # For some analytics, NaN/inf are natural; e.g. uncased languages with no true-case targets.
            if precision.isnan():
                precision = torch.zeros_like(precision)
            if recall.isnan():
                recall = torch.zeros_like(recall)
            if f1.isnan():
                f1 = torch.zeros_like(f1)
            self.log(f"val_{language}/{analytic}_precision", precision)
            self.log(f"val_{language}/{analytic}_recall", recall)
            self.log(f"val_{language}/{analytic}_f1", f1)
            if language in self._log_val_metrics_for:
                logging.info(f"{analytic} report for '{language}': {report}")

    # TODO re-enable these, in the case of using only one language.
    #   When using multiple data loaders, uncommenting these will break eval.
    #   When using one data loader, commenting these will break eval.
    # def validation_epoch_end(self, outputs) -> None:
    #     # Always use multi implementation and just use index 0.
    #     self.multi_validation_epoch_end(outputs=outputs, dataloader_idx=0)
    #
    def test_epoch_end(self, outputs) -> None:
        # Compute, reset, and log the precision/recall/f1 for punct/cap/seg for this threshold
        for analytic in ["punct_pre", "punct_post", "cap", "seg"]:
            precision, recall, f1, report = self._test_metrics[f"{analytic}_report"].compute()
            self._test_metrics[f"{analytic}_report"].reset()
            self.log(f"test_{analytic}_precision", precision)
            self.log(f"test_{analytic}_recall", recall)
            self.log(f"test_{analytic}_f1", f1)
            logging.info(f"{analytic} test report: {report}")

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0) -> None:
        self._multi_eval_epoch_end(dataloader_idx)

    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        self._eval_step(batch=batch, dataloader_idx=dataloader_idx)

    def test_step(self, batch: Tuple[torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        self._test_step(batch=batch, dataloader_idx=dataloader_idx)

    def on_validation_epoch_start(self) -> None:
        """
        This function re-seeds the validation data loaders at the beginning of each epoch in the event of using 0
        num_workers. Normally, worker_init_fn ensures that during each validation epoch the same examples are generated.
        This is needed when using no workers, and the data is loaded in the main process without calling worker_init_fn.
        """
        super().on_validation_epoch_start()
        for dataloader in self._validation_dl:
            if dataloader.num_workers == 0:
                dataloader.dataset.worker_init_fn(worker_id=0)

    @property
    def input_module(self):
        return self

    @property
    def output_module(self):
        return self

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(("B", "T"), ChannelType()),
            "lengths": NeuralType(("B",), LengthsType()),
            "punc_mask": NeuralType(("B", "T", "D"), MaskType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "punct_pre_logits": NeuralType(("B", "T", "C"), LogitsType()),
            "punct_post_logits": NeuralType(("B", "T", "C"), LogitsType()),
            "cap_logits": NeuralType(("B", "T", "D"), LogitsType()),  # D == max_subword_length
            "seg_logits": NeuralType(("B", "T", "C"), LogitsType()),  # C == 2
        }

    def input_example(
        self, min_batch_size: int = 2, max_batch_size: int = 32, min_seq_length: int = 20, max_seq_length: int = 128,
    ):
        batch_size = torch.randint(low=min_batch_size, high=max_batch_size, size=[1]).item()
        seq_length = torch.randint(low=min_seq_length, high=max_seq_length, size=[1]).item()
        input_ids = torch.randint(size=[batch_size, seq_length], low=0, high=1000)
        lengths = torch.randint(low=0, high=seq_length, size=[batch_size])
        lengths[0] = seq_length
        punc_mask = torch.randint(low=0, high=2, size=[batch_size, seq_length, self._max_token_len]).bool()
        return input_ids, lengths, punc_mask

    @typecheck()
    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor, punc_mask: torch.Tensor):
        """Intended for inference. For training/evaluation, use `run_step`"""
        seq_mask = input_ids.ne(self.tokenizer.pad_id)
        # [B, T, D]
        encoded = self.bert_model(input_ids=input_ids, attention_mask=seq_mask, token_type_ids=None,)
        # Some LMs will return a tuple
        if isinstance(encoded, tuple):
            encoded = encoded[0]

        pre_logits, post_logits, cap_logits, seg_logits = self._decoder(encoded=encoded, mask=seq_mask)

        # [B, T, max_token_len]
        cap_probs = cap_logits.sigmoid()
        # [B, T, 2]
        seg_probs = seg_logits.softmax(dim=-1)[..., 1]
        # Select the highest-scoring value.
        pre_probs = pre_logits.softmax(dim=-1)
        post_probs = post_logits.softmax(dim=-1)
        return pre_probs, post_probs, cap_probs, seg_probs

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None

    def predict_dataloader(self, config: Union[Dict, DictConfig]):
        dataset: InferencePunctCapSegDataset = InferencePunctCapSegDataset(
            tokenizer=self.tokenizer,
            input_file=config.get("input_file"),
            input_texts=config.get("texts"),
            max_length=config.get("max_length", self._max_length),
            fold_overlap=config.get("fold_overlap", 16),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=config.get("batch_size", 16),
            num_workers=config.get("num_workers", 0),
            pin_memory=config.get("pin_memory", False),
            drop_last=config.get("drop_last", False),
        )
        return dataloader

    def _unfold_tensors(
        self,
        folded_tensor: torch.Tensor,
        lengths: torch.Tensor,
        batch_ids: torch.Tensor,
        overlap: int,
        use_scale: Optional[bool] = None,
        time_dim: int = 1,
    ) -> List[torch.Tensor]:
        # Move everything to CPU
        folded_tensor = folded_tensor.cpu()
        lengths = lengths.cpu()
        batch_ids = batch_ids.cpu()
        # Use a scale by default if the input is float (and presumably probabilities)
        if use_scale is None:
            use_scale = folded_tensor.is_floating_point()
        scale: Optional[torch.Tensor] = None
        if use_scale:
            scale = torch.linspace(0, np.pi / 2, overlap).cos().unsqueeze(0)
            if scale.dim() < folded_tensor.dim():
                scale = scale.unsqueeze(-1)

        unfolded_outputs: List = []
        # For each unfolded batch index, find all subsegments
        batch_id_to_indices: DefaultDict[int, List[int]] = defaultdict(list)
        for position, batch_id in enumerate(batch_ids.tolist()):
            batch_id_to_indices[batch_id].append(position)
        # Recombine the segments of each batch element
        for batch_index, indices in batch_id_to_indices.items():
            unfolded_tensor: Optional[torch.Tensor] = None
            for index in indices:
                length = lengths[index]
                # Strip BOS and EOS here. Reduce time dim by 1 because we extract batch element.
                subsegment_tensor = folded_tensor[index].narrow(time_dim - 1, 1, length - 2)
                if unfolded_tensor is None:
                    unfolded_tensor = subsegment_tensor
                else:
                    if use_scale:
                        # Scale final segment of existing tensor
                        unfolded_tensor[-overlap:] = unfolded_tensor[-overlap:] * scale
                        # Scale beginning segment of new tensor
                        subsegment_tensor[:overlap] = subsegment_tensor[:overlap] * scale.flip(1)
                    unfolded_tensor = torch.cat(
                        (
                            unfolded_tensor.narrow(time_dim - 1, 0, unfolded_tensor.shape[time_dim - 1] - overlap // 2),
                            subsegment_tensor.narrow(time_dim - 1, overlap // 2, length - 2 - overlap // 2),
                        )
                    )
            # Always return lists, because this function is used after running model
            unfolded_outputs.append(unfolded_tensor)
        return unfolded_outputs

    def _get_char_cap_preds(self, tokens: List[str], token_preds: List[List[int]]) -> List[int]:
        """Gathers character-level truecase predictions from subword predictions"""
        char_preds: List[int] = []
        for token_num, token in enumerate(tokens):
            start = 0
            if self._using_sp:
                if token.startswith("▁"):
                    start = 1
            elif token.startswith("##"):
                start = 2
            for char_num in range(start, len(token)):
                pred = token_preds[token_num][char_num]
                char_preds.append(pred)
        return char_preds

    def _get_char_seg_preds(self, tokens: List[str], token_preds: List[int]) -> List[int]:
        """Gathers character-level sentence boundary predictions from subword predictions"""
        char_preds: List[int] = []
        current_char = 0
        for token_num, token in enumerate(tokens):
            # Find out how many input chars this subtoken consumes
            token_len = len(token)
            if self._using_sp:
                if token.startswith("▁"):
                    token_len = len(token) - 1
            else:
                if token.startswith("##"):
                    token_len = len(token) - 2
            # Advance to the end of this char
            current_char += token_len
            if token_preds[token_num] == 1:
                char_preds.append(current_char)
        return char_preds

    def _get_char_punct_preds(self, tokens: List[str], token_preds: List[int], post: bool) -> Dict[int, str]:
        """Gathers subtoken-level pre-punctuation predictions"""
        # one per character, for alignment purposes. Since these are mostly null, use a dict to specify non-null indices
        output_preds: Dict[int, str] = {}
        current_char = 0
        for token_num, token in enumerate(tokens):
            # Find out how many input chars this subtoken consumes
            token_len = len(token)
            if self._using_sp:
                if token.startswith("▁"):
                    token_len = len(token) - 1
            else:
                if token.startswith("##"):
                    token_len = len(token) - 2
            pred = token_preds[token_num]
            if post:
                label = self._punct_post_labels[pred]
            else:
                label = self._punct_pre_labels[pred]
            if post:
                current_char += token_len
            if label != NULL_PUNCT_TOKEN:
                output_preds[current_char - (1 if post else 0)] = label
            if not post:
                current_char += token_len
        return output_preds

    @torch.inference_mode()
    def infer(
        self, texts: List[str], fold_overlap: int = 20, batch_size: int = 32, max_length: Optional[int] = None,
    ) -> List[List[str]]:
        in_mode = self.training
        self.eval()
        # Default to this model's values
        if max_length is None:
            max_length = self._max_length
        dataloader = self.predict_dataloader(
            {"texts": texts, "max_length": max_length, "fold_overlap": fold_overlap, "batch_size": batch_size,}
        )
        out_texts: List[List[str]] = []
        for batch in dataloader:
            folded_input_ids, folded_batch_indices, lengths = batch
            mask = folded_input_ids.ne(self.tokenizer.pad_id)
            # [B, T, D]
            encoded = self.bert_model(input_ids=folded_input_ids, attention_mask=mask, token_type_ids=None,)
            # Some LMs will return a tuple
            if isinstance(encoded, tuple):
                encoded = encoded[0]

            pre_logits, post_logits, cap_logits, seg_logits = self._decoder(encoded=encoded, mask=mask)

            # [B, T, max_token_len]
            cap_probs = cap_logits.sigmoid()
            cap_probs_list: List[torch.Tensor] = self._unfold_tensors(
                folded_tensor=cap_probs, lengths=lengths, overlap=fold_overlap, batch_ids=folded_batch_indices
            )

            # [B, T, 2]
            seg_probs = seg_logits.softmax(dim=-1)[..., 1]
            seg_probs_list: List[torch.Tensor] = self._unfold_tensors(
                folded_tensor=seg_probs, lengths=lengths, overlap=fold_overlap, batch_ids=folded_batch_indices
            )

            # Select the highest-scoring value. Set null index to very small value (in case it was 1.0)
            pre_probs = pre_logits.softmax(dim=-1)
            pre_probs_list: List[torch.Tensor] = self._unfold_tensors(
                folded_tensor=pre_probs, lengths=lengths, overlap=fold_overlap, batch_ids=folded_batch_indices
            )

            post_probs = post_logits.softmax(dim=-1)
            post_probs_list: List[torch.Tensor] = self._unfold_tensors(
                folded_tensor=post_probs, lengths=lengths, overlap=fold_overlap, batch_ids=folded_batch_indices,
            )

            input_ids_list: List[torch.Tensor] = self._unfold_tensors(
                folded_tensor=folded_input_ids, lengths=lengths, overlap=fold_overlap, batch_ids=folded_batch_indices
            )
            for batch_idx, ids in enumerate(input_ids_list):
                ids = ids.tolist()
                tokens = self.tokenizer.ids_to_tokens(ids)
                cap_preds: List[List[int]] = cap_probs_list[batch_idx].gt(0.5).tolist()
                seg_preds: List[int] = seg_probs_list[batch_idx].gt(0.5).tolist()
                pre_preds: List[int] = pre_probs_list[batch_idx].argmax(dim=-1).tolist()
                post_preds: List[int] = post_probs_list[batch_idx].argmax(dim=-1).tolist()

                input_text = self.tokenizer.ids_to_text(ids)
                cap_char_preds = self._get_char_cap_preds(tokens=tokens, token_preds=cap_preds)
                post_tokens = self._get_char_punct_preds(tokens=tokens, token_preds=post_preds, post=True)
                pre_tokens = self._get_char_punct_preds(tokens=tokens, token_preds=pre_preds, post=False)
                break_points = self._get_char_seg_preds(tokens=tokens, token_preds=seg_preds)

                segmented_texts: List[str] = []
                output_chars: List[str] = []
                # All character-level predictions align to non-whitespace inputs
                non_whitespace_index = 0
                for input_char in list(input_text):
                    if input_char == " ":
                        output_chars.append(" ")
                        continue
                    # Maybe add punctuation before this char
                    if non_whitespace_index in pre_tokens:
                        pre_token = pre_tokens[non_whitespace_index]
                        output_chars.append(pre_token)
                    # Append true-cased input char to output
                    if cap_char_preds[non_whitespace_index] == 1:
                        output_chars.append(input_char.upper())
                    else:
                        output_chars.append(input_char.lower())
                    # Maybe add punctuation after this char
                    if non_whitespace_index in post_tokens:
                        post_token = post_tokens[non_whitespace_index]
                        output_chars.append(post_token)
                    # Maybe split sentence on this char
                    if break_points and break_points[0] == non_whitespace_index + 1:
                        segmented_texts.append("".join(output_chars).strip())
                        output_chars = []
                        del break_points[0]
                    non_whitespace_index += 1
                if output_chars:
                    out_text = "".join(output_chars).strip()
                    segmented_texts.append(out_text)
                out_texts.append(segmented_texts)
        self.train(in_mode)
        return out_texts
