import os

import hydra
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from transformers import BartTokenizerFast
from watermark import watermark

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("start1")


@hydra.main(version_base="1.1", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(watermark(packages="torch,lightning,transformers,wandb", python=True))

    # load the config file
    config = OmegaConf.create(OmegaConf.to_container(cfg))
    print(OmegaConf.to_yaml(config))

    # Load the tokenizer for mRNA
    tokenizer = BartTokenizerFast.from_pretrained(config.data.tokenizer_path)

    # Load dataset
    dataset = load_from_disk(config.data.dataset_path)

    print("END")


if __name__ == "__main__":
    main()
