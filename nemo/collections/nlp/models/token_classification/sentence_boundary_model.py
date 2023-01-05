import os
from typing import Union, Dict, Optional, List, Tuple

from pytorch_lightning import Trainer
import torch
import torch.nn as nn
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.core.neural_types import ChannelType, LogitsType, NeuralType
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.data.token_classification.sentence_boundary_dataset import (
    SentenceBoundaryDataset,
    SentenceBoundaryConfig,
)
from nemo.collections.nlp.data.token_classification.token_classification_infer_dataset import (
    TokenClassificationInferDataset,
    TokenClassificationInferDatasetConfig,
)
from nemo.core import PretrainedModelInfo, typecheck
from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.common.data import ConcatMapDataset
from nemo.utils import logging


def _worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if isinstance(dataset, ConcatMapDataset):
        sub_dataset: SentenceBoundaryDataset
        for sub_dataset in dataset.datasets:
            sub_dataset.worker_init_fn(worker_id)
    else:
        dataset.worker_init_fn(worker_id)


class SentenceBoundaryDetectionModel(NLPModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer, no_lm_init=False)
        self._loss = CrossEntropyLoss(logits_ndim=3, weight=cfg.get("loss_weight"))
        self._ignore_idx: int = cfg.get("ignore_idx", -100)
        self._decoder: TokenClassifier = TokenClassifier(
            hidden_size=self.hidden_size,
            use_transformer_init=cfg.decoder.get("use_transformer_init", True),
            num_layers=cfg.decoder.get("num_layers", 1),
            dropout=cfg.decoder.get("dropout", 0.1),
            activation="relu",
            log_softmax=False,
            num_classes=2,
        )
        self._dev_metrics = self._setup_metrics(len(self._validation_dl)) if self._validation_dl is not None else None

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            # Use `ChannelType` to make `BertModel` happy (actually is a `TokenType`)
            "input_ids": NeuralType(("B", "T"), ChannelType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "logits": NeuralType(("B", "T", "D"), LogitsType()),
        }

    def input_example(
        self, min_batch_size: int = 2, max_batch_size: int = 8, min_seq_length: int = 5, max_seq_length: int = 16
    ):
        p = next(self.parameters())
        batch_size = torch.randint(low=min_batch_size, high=max_batch_size, size=[1]).item()
        seq_length = torch.randint(low=min_seq_length, high=max_seq_length, size=[1]).item()
        input_ids = torch.randint(low=0, high=self.tokenizer.vocab_size, size=[batch_size, seq_length], device=p.device)
        return input_ids

    @classmethod
    def list_available_models(cls) -> Optional[List[PretrainedModelInfo]]:
        return None

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        pass

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self.setup_validation_data(val_data_config)

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self._validation_dl: List[torch.utils.data.DataLoader] = []
        self._validation_names: List[str] = []
        for lang, lang_config in val_data_config.data.items():
            with open_dict(lang_config):
                for k, v in val_data_config.common.items():
                    if k not in lang_config:
                        lang_config[k] = v
                lang_config.language = lang
            lang_config = SentenceBoundaryConfig(**lang_config, tokenizer=self.tokenizer)
            dataset: SentenceBoundaryDataset = SentenceBoundaryDataset(config=lang_config)
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                collate_fn=dataset.collate_fn,
                batch_size=val_data_config.get("batch_size", 128),
                num_workers=val_data_config.get("num_workers", min(8, os.cpu_count() - 1)),
                pin_memory=val_data_config.get("pin_memory", False),
                worker_init_fn=_worker_init_fn,
                persistent_workers=False,
            )
            self._validation_dl.append(dataloader)
            self._validation_names.append(f"val_{lang}")

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        datasets: List[SentenceBoundaryDataset] = []
        for lang, lang_config in train_data_config.data.items():
            print(f"Set up dataset for {lang}")
            with open_dict(lang_config):
                for k, v in train_data_config.common.items():
                    if k not in lang_config:
                        lang_config[k] = v
                lang_config.language = lang
            lang_config = SentenceBoundaryConfig(**lang_config, tokenizer=self.tokenizer)
            dataset: SentenceBoundaryDataset = SentenceBoundaryDataset(config=lang_config)
            datasets.append(dataset)
        print(f"Build concat dataset")
        concat_dataset: ConcatMapDataset = ConcatMapDataset(
            datasets=datasets, sampling_technique="temperature", sampling_temperature=5
        )
        print(f"Instantiate data loader")
        dataloader: DataLoader = DataLoader(
            dataset=concat_dataset,
            shuffle=train_data_config.get("shuffle", True),
            batch_size=train_data_config.get("batch_size", 128),
            num_workers=train_data_config.get("num_workers", os.cpu_count() - 1),
            collate_fn=datasets[0].collate_fn,
            worker_init_fn=_worker_init_fn,
        )
        print("ok")
        self._train_dl = dataloader

    def _setup_metrics(self, num_dl: int = 1) -> nn.ModuleList:
        """Creates metrics for each data loader. Typically, we have one DL per language.

        Returns:
            A :class:``nn.ModuleList``, with one element per data loader. Each element is another
            :class:``nn.ModuleList`` of metrics for that language.

        """
        module_list: nn.ModuleList = nn.ModuleList()
        for _ in range(num_dl):
            metrics: nn.ModuleDict = nn.ModuleDict(
                {
                    "loss": GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True),
                    "report": ClassificationReport(
                        num_classes=2, label_ids={"NOSTOP": 0, "FULLSTOP": 1}, mode="macro", dist_sync_on_step=False
                    ),
                }
            )
            module_list.append(metrics)
        return module_list

    @property
    def output_names(self):
        return ["probs"]

    @property
    def input_module(self):
        return self

    @typecheck()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(self.tokenizer.pad_id)
        # [B, T] -> [B, T, D]
        encoded = self.bert_model(input_ids=input_ids, attention_mask=mask, token_type_ids=None)
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        # [B, T, D] -> [B, T, C]
        logits = self._decoder(hidden_states=encoded)
        return logits

    def forward_for_export(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(self.tokenizer.pad_id)
        # [B, T] -> [B, T, D]
        encoded = self.bert_model(input_ids=input_ids, attention_mask=mask, token_type_ids=None)
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        # [B, T, D] -> [B, T, C]
        logits = self._decoder(hidden_states=encoded)
        probs = logits.softmax(-1)
        probs = probs[..., 1]
        return probs

    def training_step(self, batch, batch_idx: int):
        input_ids, targets, lengths = batch
        logits = self.forward(input_ids=input_ids)
        loss = self._loss(logits=logits, labels=targets)
        lr = self._optimizer.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)
        self.log("train_loss", loss)
        return loss

    def _eval_step(self, batch: Tuple, dataloader_idx: int = 0) -> None:
        input_ids, targets, lengths = batch
        logits = self.forward(input_ids=input_ids)
        loss = self._loss(logits=logits, labels=targets)
        # log probs are [B, T, D]
        preds = logits.argmax(dim=-1)
        mask = targets.ne(self._ignore_idx)
        metrics: nn.ModuleDict = self._dev_metrics[dataloader_idx]
        metrics["loss"](loss=loss, num_measurements=lengths.sum())
        metrics["report"](preds[mask], targets[mask])

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0) -> None:
        self._multi_eval_epoch_end(dataloader_idx)

    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        self._eval_step(batch=batch, dataloader_idx=dataloader_idx)

    def _multi_eval_epoch_end(self, dataloader_idx: int):
        """ Epoch end logic for both validation and test """
        metric_dict: nn.ModuleDict = self._dev_metrics[dataloader_idx]
        # Resolve language for better logging
        language = self._validation_dl[dataloader_idx].dataset.language

        # Compute, reset, and log the loss for this language
        loss = metric_dict["loss"].compute()
        metric_dict["loss"].reset()
        self.log(f"val_{language}_loss", loss)

        # Compute, reset, and log the precision/recall/f1 for punct/cap/seg for this language
        precision, recall, f1, report = metric_dict["report"].compute()
        metric_dict["report"].reset()
        self.log(f"val_{language}_precision", precision)
        self.log(f"val_{language}_recall", recall)
        self.log(f"val_{language}_f1", f1)
        logging.info(f"Report for '{language}': {report}")

    @torch.inference_mode()
    def infer(self, inputs: List[str], threshold: float = 0.5) -> List[List[str]]:
        self.eval()
        config: TokenClassificationInferDatasetConfig = TokenClassificationInferDatasetConfig(
            tokenizer=self.tokenizer, max_length=256, overlap=16, input_data=inputs
        )
        dataset: TokenClassificationInferDataset = TokenClassificationInferDataset(config=config)
        dataloader: DataLoader = DataLoader(dataset=dataset, batch_size=8, num_workers=1, collate_fn=dataset.collate_fn)
        outputs: List[List[str]] = []
        for batch in dataloader:
            input_ids, lengths = batch
            logits = self.forward(input_ids=input_ids)
            probs = logits.softmax(dim=-1)
            for i, length in enumerate(lengths):
                tokens = input_ids[i, 1 : length - 1].tolist()
                # Trim BOS/EOS
                stop_indices = probs[i, 1 : length - 1, 1].gt(threshold).nonzero().squeeze(1).tolist()
                # for idx in range(length):
                #     token = self.tokenizer.ids_to_text([input_ids[i, idx].item()])
                #     print(f"{token} {probs[i, idx, 1]:0.4f}")
                # print(f"{stop_indices=}")
                # Append the final token for completeness
                if not stop_indices or stop_indices[-1] != len(tokens) - 1:
                    stop_indices.append(len(tokens) - 1)
                # Start points are after every stop point (except final)
                start_indices = [0] + [x + 1 for x in stop_indices[:-1]]
                # print(f"{start_indices=}")
                # Split at the token boundaries and detokenize the texts
                split_ids = [tokens[start : stop + 1] for start, stop in zip(start_indices, stop_indices)]
                split_texts = [self.tokenizer.ids_to_text(ids) for ids in split_ids]
                # print(f"{split_ids=}")
                # print(f"{split_texts}")
                outputs.append(split_texts)
        return outputs
