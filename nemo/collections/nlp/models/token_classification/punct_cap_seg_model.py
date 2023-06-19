import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer
from typing import Dict, List, Optional, Tuple, Union

from nemo.collections.common.data import ConcatMapDataset
from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.nlp.data.token_classification.punct_cap_seg_dataset import (
    InferencePunctCapSegDataset,
    PunctCapSegDataset,
    NULL_PUNCT_TOKEN,
    ACRONYM_TOKEN,
)
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.token_classification import PunctCapSegDecoder
from nemo.core import PretrainedModelInfo, typecheck
from nemo.core.neural_types import ChannelType, NeuralType, LogitsType
from nemo.utils import logging


class PunctCapSegModel(NLPModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None) -> None:
        # Whether to predict "merge" tokens, which affects loss and data loaders, TODO no code before super init
        self._predict_merge = cfg.decoder.get("predict_merge", False)
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
        self._null_punct_post_index: int = self._punct_post_token_to_index[NULL_PUNCT_TOKEN]
        self._using_pre_punct = len(self._punct_pre_labels) > 0

        # Used for loss masking. Should by synchronized with data sets.
        self._ignore_idx: int = self._cfg.get("ignore_idx", -100)

        # Used for making character-level predictions with subwords (predict max_token_len per token)
        self._using_sp = True  # isinstance(self.tokenizer, SentencePieceTokenizer)
        if not self._using_sp:
            self._max_token_len = max(len(x) for x in self.tokenizer.vocab)
        else:
            # SentencePiece model - AutoTokenizer doesn't have 'vocab' attr for some SP models
            vocab_size = self.tokenizer.vocab_size
            self._max_token_len = max(len(self.tokenizer.ids_to_tokens([idx])[0]) for idx in range(vocab_size))

        # [B, T, num_pre_punct]
        self._punct_pre_loss: CrossEntropyLoss = CrossEntropyLoss(
            weight=cfg.loss.punct_pre.get("weight"),
            ignore_index=self._ignore_idx,
            logits_ndim=3,
            label_smoothing=cfg.loss.punct_pre.get("label_smoothing", 0.0),
        )
        # [B, T, num_post_punct]
        self._punct_post_loss: CrossEntropyLoss = CrossEntropyLoss(
            weight=cfg.loss.punct_post.get("weight"),
            ignore_index=self._ignore_idx,
            logits_ndim=3,
            label_smoothing=cfg.loss.punct_post.get("label_smoothing", 0.0),
        )
        # [B, T, 2] binary preds
        self._seg_loss: CrossEntropyLoss = CrossEntropyLoss(
            weight=cfg.loss.seg.get("weight"),
            ignore_index=self._ignore_idx,
            logits_ndim=3,
            label_smoothing=cfg.loss.seg.get("label_smoothing", 0.0),
        )
        # [B, T, max_chars_per_subword]
        # For true-casing, we use multi-label classification to predict for each char in a subword
        self._cap_loss: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(cfg.loss.cap["weight"]) if "weight" in cfg.loss.cap else None
        )

        self._merge_loss: Optional[CrossEntropyLoss] = None
        if self._predict_merge:
            self._merge_loss = CrossEntropyLoss(
                weight=cfg.loss.merge.get("weight"),
                ignore_index=self._ignore_idx,
                logits_ndim=3,
                label_smoothing=cfg.loss.merge.get("label_smoothing", 0.0),
            )

        # Weights can be specified in punct{-pre,-post}, cap, seg, [merge] order.
        num_loss = 5 if self._predict_merge else 4
        self._agg_loss = AggregatorLoss(num_inputs=num_loss, weights=cfg.loss.get("agg_loss_weights"))

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
            if len(self._validation_dl) == 1:
                self._dev_metrics = nn.ModuleList([self._dev_metrics])
        self._train_metrics: nn.ModuleDict = self._setup_metrics()
        # module list of module dict
        self._test_metrics: nn.ModuleDict = self._setup_metrics()

    def register_bert_model(self):
        # Base class implementation is buggy and unnecessary for non-Riva users... disable it.
        pass

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
            if self._predict_merge:
                metrics["merge_report"] = ClassificationReport(
                    num_classes=2, label_ids={"NOMERGE": 0, "MERGE": 1}, mode="macro", dist_sync_on_step=False
                )
            if self._using_pre_punct:
                metrics["punct_pre_report"] = ClassificationReport(
                    num_classes=len(self._punct_pre_labels),
                    label_ids=self._punct_pre_token_to_index,
                    mode="macro",
                    dist_sync_on_step=False,
                )

            module_list.append(metrics)
        if num_dl == 1:
            return module_list[0]
        return module_list

    def _setup_eval_dataloaders_from_config(self, cfg) -> List[torch.utils.data.DataLoader]:
        if cfg is None:
            return None
        dataloaders: List[torch.utils.data.DataLoader] = []
        for ds_config in cfg.datasets:
            # Add all common variables, if not set already
            with open_dict(ds_config):
                ds_config.predict_merge = self._predict_merge
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
            cfg.dataset.predict_merge = self._predict_merge
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
        if cfg is None:
            return None
        datasets: List[PunctCapSegDataset] = []
        for ds_config in cfg.datasets:
            # Add all common variables, if not set already
            with open_dict(ds_config):
                ds_config.predict_merge = self._predict_merge
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
        # All inputs and targets are shape [B, T]  # TODO assuming merge targets?
        (inputs, punct_pre_targets, punct_post_targets, cap_targets, seg_targets, _, merge_targets) = batch
        mask = inputs.ne(self.tokenizer.pad_id)
        # Encoded output is [B, T, D]
        encoded = self.bert_model(input_ids=inputs, attention_mask=mask, token_type_ids=None)
        # Some LMs will return a tuple
        if isinstance(encoded, tuple):
            encoded = encoded[0]

        # Make a binary mask from the post punc targets
        punc_mask = punct_post_targets.eq(self._cfg.get("ignore_idx", -100))
        punct_targets_for_decoder = punct_post_targets.masked_fill(punc_mask, self._null_punct_post_index)
        # In training mode, always feed ground truth predictions to the heads.
        # For val, it's useful to use reference to not penalize the cap/seg heads for bad punctuation predictions when
        # selecting the best model.
        punct_logits_pre, punct_logits_post, cap_logits, seg_logits, merge_logits = self._decoder(
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
        if self._predict_merge:
            merge_loss = self._merge_loss(logits=merge_logits, labels=merge_targets)
        loss = self._agg_loss.forward(
            loss_1=punct_pre_loss, loss_2=punct_post_loss, loss_3=cap_loss, loss_4=seg_loss, loss_5=merge_loss  # noqa
        )

        return loss, punct_logits_pre, punct_logits_post, cap_logits, seg_logits, merge_logits

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int):
        (loss, punct_pre_logits, punct_post_logits, cap_logits, seg_logits, merge_logits) = self._run_step(batch)
        lr = self._optimizer.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)
        self.log("train_loss", loss)

        # Maybe accumulate batch metrics
        if batch_idx % self._cfg.get("batch_metrics_every_n_steps", 10) == 0:
            _, punct_pre_targets, punct_post_targets, cap_targets, seg_targets, _, merge_targets = batch
            punct_pre_preds = punct_pre_logits.argmax(dim=-1)
            punct_post_preds = punct_post_logits.argmax(dim=-1)
            cap_mask = cap_targets.ne(self._ignore_idx)
            cap_preds = cap_logits[cap_mask].sigmoid().gt(0.5)
            seg_mask = seg_targets.ne(self._ignore_idx)
            seg_preds = seg_logits.argmax(dim=-1)
            merge_mask = merge_targets.ne(self._ignore_idx)
            merge_preds = merge_logits.argmax(dim=-1)

            punct_pre_mask = punct_pre_targets.ne(self._ignore_idx)
            punct_post_mask = punct_post_targets.ne(self._ignore_idx)
            self._train_metrics["punct_pre_report"](punct_pre_preds[punct_pre_mask], punct_pre_targets[punct_pre_mask])
            self._train_metrics["punct_post_report"](
                punct_post_preds[punct_post_mask], punct_post_targets[punct_post_mask]
            )
            self._train_metrics["cap_report"](cap_preds, cap_targets[cap_mask])
            self._train_metrics["seg_report"](seg_preds[seg_mask], seg_targets[seg_mask])
            self._train_metrics["merge_report"](merge_preds[merge_mask], merge_targets[merge_mask])

        # Maybe log and reset batch metrics
        if batch_idx > 0 and batch_idx % self._trainer.log_every_n_steps == 0:
            # todo don't need keys iterate over all metrics (currently, loss is in the dict)
            analytics = ["punct_pre", "punct_post", "cap", "seg", "merge"]
            for analytic in analytics:
                metric = self._train_metrics[f"{analytic}_report"]
                precision, recall, f1, report = metric.compute()
                metric.reset()
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
        (loss, punct_pre_logits, punct_post_logits, cap_logits, seg_logits, merge_logits) = self._run_step(batch)
        _, punct_pre_targets, punct_post_targets, cap_targets, seg_targets, _, merge_targets = batch
        # All log probs are [B, T, D]
        punct_pre_preds = punct_pre_logits.argmax(dim=-1)
        punct_post_preds = punct_post_logits.argmax(dim=-1)
        cap_mask = cap_targets.ne(self._ignore_idx)
        cap_preds = cap_logits[cap_mask].sigmoid().gt(0.5)
        seg_mask = seg_targets.ne(self._ignore_idx)
        seg_preds = seg_logits.argmax(dim=-1)
        merge_mask = merge_targets.ne(self._ignore_idx)
        merge_preds = merge_logits.argmax(dim=-1)

        metrics: nn.ModuleDict = self._dev_metrics[dataloader_idx]
        punct_pre_mask = punct_pre_targets.ne(self._ignore_idx)
        punct_post_mask = punct_post_targets.ne(self._ignore_idx)
        num_targets = punct_pre_mask.sum() + punct_post_mask.sum() + cap_mask.sum() + seg_mask.sum() + merge_mask.sum()
        metrics["punct_pre_report"](punct_pre_preds[punct_pre_mask], punct_pre_targets[punct_pre_mask])
        metrics["punct_post_report"](punct_post_preds[punct_post_mask], punct_post_targets[punct_post_mask])
        metrics["cap_report"](cap_preds, cap_targets[cap_mask])
        metrics["seg_report"](seg_preds[seg_mask], seg_targets[seg_mask])
        metrics["merge_report"](merge_preds[merge_mask], merge_targets[merge_mask])
        metrics["loss"](loss=loss, num_measurements=num_targets)

    def _test_step(self, batch: Tuple, dataloader_idx: int = 0) -> None:
        (loss, punct_pre_logits, punct_post_logits, cap_logits, seg_logits, merge_logits) = self._run_step(
            batch, testing=False
        )
        _, punct_pre_targets, punct_post_targets, cap_targets, seg_targets, _, merge_targets = batch
        # Prepare masks
        cap_mask = cap_targets.ne(self._ignore_idx)
        seg_mask = seg_targets.ne(self._ignore_idx)
        merge_mask = merge_targets.ne(self._ignore_idx)
        punct_pre_mask = punct_pre_targets.ne(self._ignore_idx)
        punct_post_mask = punct_post_targets.ne(self._ignore_idx)
        # Get all probs. All log probs are [B, T, D]
        punct_pre_probs = punct_pre_logits.softmax(dim=-1)
        punct_post_probs = punct_post_logits.softmax(dim=-1)
        seg_probs = seg_logits.softmax(dim=-1)[..., 1]
        merge_probs = merge_logits.softmax(dim=-1)[..., 1]
        cap_probs = cap_logits[cap_mask].sigmoid()

        punct_pre_preds = punct_pre_probs.argmax(dim=-1)
        punct_post_preds = punct_post_probs.argmax(dim=-1)
        seg_preds = seg_probs.ge(0.5)
        merge_preds = merge_probs.ge(0.5)
        cap_preds = cap_probs.ge(0.5)
        self._test_metrics["punct_pre_report"](punct_pre_preds[punct_pre_mask], punct_pre_targets[punct_pre_mask])
        self._test_metrics["punct_post_report"](punct_post_preds[punct_post_mask], punct_post_targets[punct_post_mask])
        self._test_metrics["cap_report"](cap_preds, cap_targets[cap_mask])
        self._test_metrics["seg_report"](seg_preds[seg_mask], seg_targets[seg_mask])
        self._test_metrics["merge_report"](merge_preds[merge_mask], merge_targets[merge_mask])

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
        analytics = ["punct_pre", "punct_post", "cap", "seg", "merge"]
        for analytic in analytics:
            metric = metric_dict[f"{analytic}_report"]
            precision, recall, f1, report = metric.compute()
            metric.reset()
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

    # FIXME if this is uncommented, using multiple val data loaders will fail. If this is commented, using only one
    #  val data loader will fail. Currently toggled manually based on the recipe being run.
    # def validation_epoch_end(self, outputs) -> None:
    #     # Always use multi implementation and just use index 0.
    #     self.multi_validation_epoch_end(outputs=outputs, dataloader_idx=0)

    def test_epoch_end(self, outputs) -> None:
        # Compute, reset, and log the precision/recall/f1 for punct/cap/seg for this threshold
        analytics = ["punct_pre", "punct_post", "cap", "seg", "merge"]
        for analytic in analytics:
            precision, recall, f1, report = self._test_metrics[f"{analytic}_report"].compute()
            self._test_metrics[f"{analytic}_report"].reset()
            self.log(f"test_{analytic}_precision", precision)
            self.log(f"test_{analytic}_recall", recall)
            self.log(f"test_{analytic}_f1", f1)
            logging.info(f"{analytic} test report: {report}")

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0) -> None:
        return self._multi_eval_epoch_end(dataloader_idx)

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
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        outputs = {
            "pre_preds": NeuralType(("B", "T", "C"), LogitsType()),
            "post_preds": NeuralType(("B", "T", "C"), LogitsType()),
            "cap_preds": NeuralType(("B", "T", "D"), LogitsType()),  # D == max_subword_length
            "seg_preds": NeuralType(("B", "T"), LogitsType()),  # C == 2
        }
        if self._predict_merge:
            outputs["merge_preds"] = (NeuralType(("B", "T"), LogitsType()),)  # C == 2
        return outputs

    def input_example(
        self, min_batch_size: int = 2, max_batch_size: int = 32, min_seq_length: int = 20, max_seq_length: int = 128,
    ):
        batch_size = torch.randint(low=min_batch_size, high=max_batch_size, size=[1]).item()
        seq_length = torch.randint(low=min_seq_length, high=max_seq_length, size=[1]).item()
        input_ids = torch.randint(size=[batch_size, seq_length], low=0, high=1000)
        # Note: we don't use lengths because it can be inferred from where the input_ids == the tokenizer pad ID
        return input_ids

    @typecheck()
    def forward(self, input_ids: torch.Tensor):
        """Intended for inference. For training/evaluation, use `run_step`"""
        seq_mask = input_ids.ne(self.tokenizer.pad_id)
        # [B, T, D]
        encoded = self.bert_model(input_ids=input_ids, attention_mask=seq_mask, token_type_ids=None,)
        # Some LMs will return a tuple
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        # TODO add merge logic
        pre_logits, post_logits, cap_logits, seg_logits = self._decoder(encoded=encoded, mask=seq_mask)

        # [B, T, max_token_len]
        cap_preds = cap_logits.sigmoid().gt(0.5)
        # [B, T]
        seg_preds = seg_logits.softmax(dim=-1)[..., 1].gt(0.05)  # TODO tmp change need to expose
        _, pre_preds = pre_logits.max(dim=-1)
        _, post_preds = post_logits.max(dim=-1)
        return pre_preds, post_preds, cap_preds, seg_preds

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None

    def predict_dataloader(self, config: Union[Dict, DictConfig]):
        dataset: InferencePunctCapSegDataset = InferencePunctCapSegDataset(
            tokenizer=self.tokenizer,
            input_file=config.get("input_file"),
            input_texts=config.get("texts"),
            max_length=config.get("max_length", self._max_length),
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
            if label == ACRONYM_TOKEN:
                # All characters in this subtoken are punctuated with a period
                for i in range(current_char - token_len, current_char):
                    output_preds[i] = "."
            elif label != NULL_PUNCT_TOKEN:
                output_preds[current_char - (1 if post else 0)] = label
            if not post:
                current_char += token_len
        return output_preds

    @torch.inference_mode()
    def infer(self, texts: List[str], batch_size: int = 32, max_length: Optional[int] = None,) -> List[List[str]]:
        in_mode = self.training
        self.eval()
        # Default to this model's values
        if max_length is None:
            max_length = self._max_length
        dataloader = self.predict_dataloader({"texts": texts, "max_length": max_length, "batch_size": batch_size,})
        out_texts: List[List[str]] = []
        for batch in dataloader:
            input_ids, lengths = batch
            mask = input_ids.ne(self.tokenizer.pad_id)
            # [B, T, D]
            encoded = self.bert_model(input_ids=input_ids, attention_mask=mask, token_type_ids=None,)
            # Some LMs will return a tuple
            if isinstance(encoded, tuple):
                encoded = encoded[0]

            pre_logits, post_logits, cap_logits, seg_logits, merge_logits = self._decoder(encoded=encoded, mask=mask)
            # [B, T, max_token_len]
            cap_probs = cap_logits.sigmoid()
            # [B, T, 2]
            seg_probs = seg_logits.softmax(dim=-1)[..., 1]
            # [B, T, 2]
            merge_probs = merge_logits.softmax(dim=-1)[..., 1]
            # Select the highest-scoring value. Set null index to very small value (in case it was 1.0)
            pre_probs = pre_logits.softmax(dim=-1)
            post_probs = post_logits.softmax(dim=-1)
            batch_size = input_ids.shape[0]
            for batch_idx in range(batch_size):
                length = lengths[batch_idx]
                # note we strip BOS/EOS in all indexing
                ids = input_ids[batch_idx, 1 : length - 1].tolist()
                tokens = self.tokenizer.ids_to_tokens(ids)
                cap_preds: List[List[int]] = cap_probs[batch_idx, 1 : length - 1].gt(0.5).tolist()
                seg_preds: List[int] = seg_probs[batch_idx, 1 : length - 1].gt(0.05).tolist()
                merge_preds: List[int] = merge_probs[batch_idx, 1 : length - 1].gt(0.5).tolist()
                pre_preds: List[int] = pre_probs[batch_idx, 1 : length - 1].argmax(dim=-1).tolist()
                post_preds: List[int] = post_probs[batch_idx, 1 : length - 1].argmax(dim=-1).tolist()

                input_text = self.tokenizer.ids_to_text(ids)
                cap_char_preds = self._get_char_cap_preds(tokens=tokens, token_preds=cap_preds)
                post_tokens = self._get_char_punct_preds(tokens=tokens, token_preds=post_preds, post=True)
                pre_tokens = self._get_char_punct_preds(tokens=tokens, token_preds=pre_preds, post=False)
                break_points = self._get_char_seg_preds(tokens=tokens, token_preds=seg_preds)
                merge_points = self._get_char_seg_preds(tokens=tokens, token_preds=merge_preds)

                segmented_texts: List[str] = []
                output_chars: List[str] = []
                # All character-level predictions align to non-whitespace inputs
                non_whitespace_index = 0
                for input_char in list(input_text):
                    if input_char == " ":
                        # Skip this space if the previous subtoken should "merge" with the next
                        if merge_points and merge_points[0] == non_whitespace_index:
                            del merge_points[0]
                            continue
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
