import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models import PunctCapSegModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="punct_cap_seg_local")
def main(cfg: DictConfig) -> None:
    # torch.manual_seed(42)
    print(cfg)
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    logging.info(f"Config: {OmegaConf.to_yaml(cfg)}")
    model: PunctCapSegModel = PunctCapSegModel(cfg.model, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg)
    # model.bert_model.freeze()
    # model.save_to("/Users/shane/models/punct_cap_seg_512.nemo")
    # print(model.bert_model.config)
    # sys.exit()
    # model.freeze()
    # model._decoder._punct_head_pre.unfreeze()
    # model.bert_model.requires_grad_(False)
    # model._decoder._punct_head_post.requires_grad_(False)
    # model._decoder._punct_head_pre.requires_grad_(False)
    trainer.fit(model)


if __name__ == "__main__":
    main()
