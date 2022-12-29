import torch

from nemo.collections.common.tokenizers import SentencePieceTokenizer
from nemo.collections.common.data import ConcatMapDataset
from nemo.collections.nlp.data.token_classification.sentence_boundary_dataset import (
    SentenceBoundaryConfig,
    SentenceBoundaryDataset,
)


def main():
    tok: SentencePieceTokenizer = SentencePieceTokenizer(model_path="/Users/shane/corpora/wmt/spe.model")

    en_cfg = SentenceBoundaryConfig(
        tokenizer=tok,
        language="en",
        text_files="/Users/shane/corpora/wmt/processed/en.train.txt",
        max_length=128,
        min_concat=0,
        max_concat=3,
        p_lowercase=0.5,
        target_pad_id=-100,
        continuous_script=False,
        whitespace_fullstops=False,
        seed=None,
    )
    es_cfg = SentenceBoundaryConfig(
        tokenizer=tok,
        language="es",
        text_files="/Users/shane/corpora/wmt/processed/es.train.txt",
        max_length=128,
        min_concat=0,
        max_concat=3,
        p_lowercase=0.5,
        target_pad_id=-100,
        continuous_script=False,
        whitespace_fullstops=False,
        seed=None,
    )
    zh_cfg = SentenceBoundaryConfig(
        tokenizer=tok,
        language="zh",
        text_files="/Users/shane/corpora/wmt/processed/zh.train.txt",
        max_length=128,
        min_concat=1,
        max_concat=3,
        p_lowercase=0.5,
        target_pad_id=-100,
        continuous_script=True,
        seed=None,
    )

    print("Instantaiate en")
    english_ds: SentenceBoundaryDataset = SentenceBoundaryDataset(config=en_cfg)
    print("Instantaiate es")
    spanish_ds: SentenceBoundaryDataset = SentenceBoundaryDataset(config=es_cfg)
    print("Instantaiate zh")
    chinese_ds: SentenceBoundaryDataset = SentenceBoundaryDataset(config=zh_cfg)
    print("create concate")
    concat_ds = ConcatMapDataset(datasets=[english_ds, spanish_ds])
    print("ok")
    dl = torch.utils.data.DataLoader(
        dataset=concat_ds,
        batch_size=2,
        num_workers=2,
        shuffle=True,
        collate_fn=concat_ds.datasets[0].collate_fn,
        worker_init_fn=concat_ds.datasets[0].worker_init_fn,
    )
    input_ids, targets = chinese_ds[0]
    print(tok.ids_to_tokens(input_ids.tolist()))
    print(f"{input_ids.tolist()=}")
    print(f"{targets.tolist()=}")
    print(tok.ids_to_text(input_ids.tolist()))
    input_ids, targets = chinese_ds[1]
    print(tok.ids_to_text(input_ids.tolist()))
    #
    # for i, batch in enumerate(dl):
    #     batch_input_ids, batch_targets, batch_lens = batch
    #     # print(batch_input_ids)
    #     # print(batch_targets)
    #     # print(batch_lens)
    #     if i == 3:
    #         break


if __name__ == "__main__":
    main()
