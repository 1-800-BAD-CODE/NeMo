from typing import List

from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models.token_classification.sentence_boundary_model import SentenceBoundaryDetectionModel
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="conf", config_name="sentence_boundary_detection_infer")
def main(cfg: DictConfig) -> None:
    logging.info(f"Config: {OmegaConf.to_yaml(cfg)}")
    model = SentenceBoundaryDetectionModel.restore_from(cfg.model)
    outputs: List[List[str]] = model.infer(inputs=cfg.input_data, threshold=cfg.threshold)
    for batch_idx, split_texts in enumerate(outputs):
        print(f"Outputs for input {batch_idx}:")
        for text_num, text in enumerate(split_texts):
            print(f"\t{text}")


if __name__ == "__main__":
    main()
