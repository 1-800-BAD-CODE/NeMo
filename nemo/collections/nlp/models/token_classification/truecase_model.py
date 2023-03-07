import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer
from typing import Dict, List, Optional, Tuple, Union

from nemo.collections.common.data import ConcatMapDataset
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.common.tokenizers import SentencePieceTokenizer
from nemo.collections.nlp.data.token_classification.truecase_dataset import TruecaseDataset, InferenceTruecaseDataset
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.token_classification.punct_cap_seg_modules import ClassificationHead
from nemo.core import PretrainedModelInfo, typecheck
from nemo.core.neural_types import ChannelType, NeuralType, LogitsType
from nemo.utils import logging


class TruecaseModel(NLPModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None) -> None:
        super().__init__(cfg=cfg, trainer=trainer)
        # During training, print metrics for these languages only
        self._log_val_metrics_for = set(cfg.get("log_val_metrics_for", []))
        # Whether to print the metrics report for training data. Generates a lot of output.
        self._log_train_metrics = cfg.get("log_train_metrics", False)
        # Should be set to the training DS max length; default to the positional embeddings size
        self._max_length = self._cfg.get("max_length", self.bert_model.config.max_position_embeddings)

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

        # [B, T, max_chars_per_subword]
        # Use multi-label classification to predict for each char in a subword
        self._loss: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(cfg.loss["weight"]) if "weight" in cfg.loss else None
        )

        with open_dict(self._cfg.decoder):
            self._cfg.decoder.punct_num_classes_post = len(self._punct_post_labels)
            self._cfg.decoder.punct_num_classes_pre = len(self._punct_pre_labels)
            self._cfg.decoder.max_subword_length = self._max_token_len
        self._decoder: ClassificationHead = TruecaseModel.from_config_dict(self._cfg.decoder)

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

    def register_bert_model(self):
        # Base class implementation is buggy and unnecessary for non-Riva users... disable it.
        pass

    def maybe_init_from_pretrained_checkpoint(self, cfg: DictConfig, map_location: str = "cpu"):
        if cfg.get("init_from_nemo_model") is not None:
            restored_model: TruecaseModel
            with open_dict(cfg):
                if isinstance(cfg.init_from_nemo_model, str):
                    model_path = cfg.init_from_nemo_model
                    restored_model = self.restore_from(
                        model_path, map_location=map_location, strict=cfg.get("init_strict", True)  # noqa
                    )
                    self.load_state_dict(restored_model.state_dict(), strict=False)
                    logging.info(f"Model checkpoint restored from nemo file with path : `{model_path}`")
                elif isinstance(cfg.init_from_nemo_model, (DictConfig, dict)):
                    for model_load_cfg in cfg.init_from_nemo_model.values():
                        model_path = model_load_cfg.path
                        restored_model = self.restore_from(
                            model_path, map_location=map_location, strict=cfg.get("init_strict", True)  # noqa
                        )
                        self.load_part_of_state_dict(
                            state_dict=restored_model.state_dict(),
                            include=model_load_cfg.pop("include", [""]),
                            exclude=model_load_cfg.pop("exclude", []),
                            load_from_string=f"nemo file with path `{model_path}`",
                        )
                else:
                    raise TypeError("Invalid type: init_from_nemo_model is not a string or a dict!")
            # The rest of this function restores weights for certain layers that can change shape (embs, labels) or
            # be shuffled (vocab). It's a little hacky and presumptive.
            with torch.inference_mode():
                # Mess with the embeddings
                """
                >>> m.bert_model.embeddings
                BertEmbeddings(
                  (word_embeddings): Embedding(64000, 512, padding_idx=0)
                  (position_embeddings): Embedding(128, 512)
                  (token_type_embeddings): Embedding(1, 512)
                  (LayerNorm): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                """
                # Allow a modified vocab, while preserving embeddings for known tokens
                if self.tokenizer.vocab != restored_model.tokenizer.vocab:
                    num_embeddings_restored = 0
                    old_token_to_idx: Dict[str, int] = {
                        token: idx for idx, token in enumerate(restored_model.tokenizer.vocab)
                    }
                    for i, token in self.tokenizer.vocab:
                        # We'll try to find this token, or a similar token, in the old vocab
                        old_idx: Optional[int]
                        if token in old_token_to_idx:
                            # Exact token match
                            old_idx = old_token_to_idx[token]
                        elif token.startswith("▁"):
                            # See if BOW token exists as an inter-word piece
                            old_idx = old_token_to_idx.get(token[1:])
                        else:
                            # Check if token exists as BOW piece
                            old_idx = old_token_to_idx.get(f"▁{token}")
                        if old_idx is not None:
                            self.bert_model.embeddings.word_embeddings.weight[
                                old_idx
                            ] = restored_model.bert_model.embeddings.word_embeddings.weight[old_idx]
                            num_embeddings_restored += 1
                    vocab_size = len(self.tokenizer.vocab)
                    pct = num_embeddings_restored / vocab_size
                    logging.info(f"Restored {num_embeddings_restored} of {vocab_size} ({pct:0.2%}) of token embeddings")
                # Preserve positional embeddings when changing max length.
                restored_max_len = restored_model.bert_model.embeddings.position_embeddings.weight.shape[0]
                new_max_len = self.bert_model.embeddings.position_embeddings.weight.shape[0]
                if new_max_len != restored_max_len:
                    logging.info(f"Restoring positional embeddings from {restored_max_len} to {new_max_len}")
                    for i in range(new_max_len):
                        # If old model had fewer positions, extend the final embedding
                        old_idx = min(i, restored_max_len - 1)
                        self.bert_model.embeddings.position_embeddings.weight[
                            i
                        ] = restored_model.bert_model.embeddings.position_embeddings.weight[old_idx].clone()
            del restored_model

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
                    "report": ClassificationReport(
                        num_classes=2, label_ids={"LOWER": 0, "UPPER": 1}, mode="macro", dist_sync_on_step=False
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
            dataset: TruecaseDataset = instantiate(ds_config)
            if not isinstance(dataset, TruecaseDataset):
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
        dataset: TruecaseDataset = instantiate(cfg.dataset)
        if not isinstance(dataset, TruecaseDataset):
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
        datasets: List[TruecaseDataset] = []
        for ds_config in cfg.datasets:
            # Add all common variables, if not set already
            with open_dict(ds_config):
                for k, v in cfg.get("common", {}).items():
                    if k not in ds_config:
                        ds_config[k] = v
            dataset: TruecaseDataset = instantiate(ds_config)
            if not isinstance(dataset, TruecaseDataset):
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

    def _run_step(self, input_ids: torch.Tensor, targets: torch.Tensor):
        # All inputs and targets are shape [B, T]
        mask = input_ids.ne(self.tokenizer.pad_id)
        # Encoded output is [B, T, D]
        encoded = self.bert_model(input_ids=input_ids, attention_mask=mask, token_type_ids=None)
        # Some LMs will return a tuple
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        logits = self._decoder(encoded)

        # If all elements are uncased, BCE returns nan. So set to zero if no targets (ja, zh, hi, etc.).
        mask = targets.ne(self._ignore_idx)
        if mask.any():
            loss = self._loss(input=logits[mask], target=targets[mask].float())
        else:
            # Dimensionless 0.0 like logits
            loss = logits.new_zeros(1).squeeze()

        return loss, logits

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int):
        loss, logits = self._run_step(batch)
        lr = self._optimizer.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)
        self.log("train_loss", loss)

        # Maybe accumulate batch metrics
        if batch_idx % self._cfg.get("batch_metrics_every_n_steps", 10) == 0:
            input_ids, targets, _ = batch
            loss, logits = self._run_step(input_ids, targets)
            mask = targets.ne(self._ignore_idx)
            preds = logits[mask].sigmoid().gt(0.5)
            self._train_metrics["report"](preds, targets[mask])

        # Maybe log and reset batch metrics
        if batch_idx > 0 and batch_idx % self._trainer.log_every_n_steps == 0:
            precision, recall, f1, report = self._train_metrics["report"].compute()
            self._train_metrics["report"].reset()
            # For some analytics, NaN/inf are natural; e.g. uncased languages with no true-case targets.
            if precision.isnan():
                precision = torch.zeros_like(precision)
            if recall.isnan():
                recall = torch.zeros_like(recall)
            if f1.isnan():
                f1 = torch.zeros_like(f1)
            self.log("train_batch/precision", precision)
            self.log("train_batch/recall", recall)
            self.log("train_batch/f1", f1)
            if self._log_train_metrics:
                logging.info(f"Train batch report: {report}")

        return loss

    def _eval_step(self, batch: Tuple, dataloader_idx: int = 0) -> None:
        input_ids, targets, _ = batch
        loss, logits = self._run_step(batch)
        lr = self._optimizer.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)
        self.log("train_loss", loss)

        mask = targets.ne(self._ignore_idx)
        preds = logits[mask].sigmoid().gt(0.5)

        metrics: nn.ModuleDict = self._dev_metrics[dataloader_idx]
        num_targets = mask.sum()
        metrics["loss"](loss=loss, num_measurements=num_targets)
        metrics["report"](preds, targets[mask])

    def _test_step(self, batch: Tuple, dataloader_idx: int = 0) -> None:
        input_ids, targets, _ = batch
        loss, logits = self._run_step(batch)
        lr = self._optimizer.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)
        self.log("train_loss", loss)

        mask = targets.ne(self._ignore_idx)
        preds = logits[mask].sigmoid().gt(0.5)

        num_targets = mask.sum()
        self._test_metrics["loss"](loss=loss, num_measurements=num_targets)
        self._test_metrics["report"](preds, targets[mask])

    def _get_language_for_dl_idx(self, idx: int) -> str:
        ds: TruecaseDataset = self._validation_dl[idx].dataset
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

        # Compute, reset, and log the precision/recall/f1 for this language
        precision, recall, f1, report = metric_dict["report"].compute()
        metric_dict["report"].reset()
        # For some analytics, NaN/inf are natural; e.g. uncased languages with no true-case targets.
        if precision.isnan():
            precision = torch.zeros_like(precision)
        if recall.isnan():
            recall = torch.zeros_like(recall)
        if f1.isnan():
            f1 = torch.zeros_like(f1)
        self.log(f"val_{language}/precision", precision)
        self.log(f"val_{language}/recall", recall)
        self.log(f"val_{language}/f1", f1)
        if language in self._log_val_metrics_for:
            logging.info(f"Val Report for '{language}': {report}")

    # TODO re-enable these, in the case of using only one language.
    #   When using multiple data loaders, uncommenting these will break eval.
    #   When using one data loader, commenting these will break eval.
    # def validation_epoch_end(self, outputs) -> None:
    #     # Always use multi implementation and just use index 0.
    #     self.multi_validation_epoch_end(outputs=outputs, dataloader_idx=0)
    #
    def test_epoch_end(self, outputs) -> None:
        # Compute, reset, and log the precision/recall/f1
        precision, recall, f1, report = self._test_metrics["report"].compute()
        self._test_metrics["report"].reset()
        self.log("test_precision", precision)
        self.log("test__recall", recall)
        self.log("test_f1", f1)
        logging.info(f"Test report: {report}")

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
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "preds": NeuralType(("B", "T", "D"), LogitsType()),  # D == max_subword_length
        }

    def input_example(
            self, min_batch_size: int = 2, max_batch_size: int = 32, min_seq_length: int = 20,
            max_seq_length: int = 128,
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
        encoded = self.bert_model(input_ids=input_ids, attention_mask=seq_mask, token_type_ids=None, )
        # Some LMs will return a tuple
        if isinstance(encoded, tuple):
            encoded = encoded[0]

        logits = self._decoder(encoded=encoded)

        # [B, T, max_token_len]
        preds = logits.sigmoid().gt(0.5)
        return preds

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None

    def predict_dataloader(self, config: Union[Dict, DictConfig]):
        dataset: InferenceTruecaseDataset = InferenceTruecaseDataset(
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

    def _get_char_preds(self, tokens: List[str], token_preds: List[List[int]]) -> List[int]:
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

    @torch.inference_mode()
    def infer(self, texts: List[str], batch_size: int = 32, max_length: Optional[int] = None, ) -> List[str]:
        in_mode = self.training
        self.eval()
        # Default to this model's values
        if max_length is None:
            max_length = self._max_length
        dataloader = self.predict_dataloader({"texts": texts, "max_length": max_length, "batch_size": batch_size, })
        out_texts: List[str] = []
        for batch in dataloader:
            input_ids, lengths = batch
            mask = input_ids.ne(self.tokenizer.pad_id)
            # [B, T, D]
            encoded = self.bert_model(input_ids=input_ids, attention_mask=mask, token_type_ids=None, )
            # Some LMs will return a tuple
            if isinstance(encoded, tuple):
                encoded = encoded[0]

            logits = self._decoder(encoded=encoded)
            # [B, T, max_token_len]
            probs = logits.sigmoid()
            for batch_idx, ids in enumerate(input_ids):
                # note we strip BOS/EOS in all indexing
                ids = ids.tolist()[1:-1]
                tokens = self.tokenizer.ids_to_tokens(ids)
                length = lengths[batch_idx]
                preds: List[List[int]] = probs[batch_idx, 1: length - 1].gt(0.5).tolist()

                input_text = self.tokenizer.ids_to_text(ids)
                char_preds = self._get_char_preds(tokens=tokens, token_preds=preds)

                output_chars: List[str] = []
                # All character-level predictions align to non-whitespace inputs
                non_whitespace_index = 0
                for input_char in list(input_text):
                    if input_char == " ":
                        output_chars.append(" ")
                        continue
                    # Append true-cased input char to output
                    if char_preds[non_whitespace_index] == 1:
                        output_chars.append(input_char.upper())
                    else:
                        output_chars.append(input_char.lower())
                    non_whitespace_index += 1
                truecased_text = "".join(output_chars)
                out_texts.append(truecased_text)
        self.train(in_mode)
        return out_texts
