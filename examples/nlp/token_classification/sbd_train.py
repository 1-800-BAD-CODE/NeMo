
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models.token_classification.sentence_boundary_model import SentenceBoundaryDetectionModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="sentence_boundary_detection")
def main(cfg: DictConfig) -> None:
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
    model = SentenceBoundaryDetectionModel(cfg.model, trainer=trainer)

    print(f"Model: {model}")
    print(f"Model params: {model.num_weights}")
    trainer.fit(model)


if __name__ == '__main__':
    main()
