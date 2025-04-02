import os
from functools import partial
from pathlib import Path

import hydra
import torch
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from transformers import (
    BartForSequenceClassification,
    BartTokenizerFast,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from watermark import watermark

import wandb
from utils import compute_multi_class_classification_metrics, compute_regression_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"




@hydra.main(version_base="1.1", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(watermark(packages="torch,lightning,transformers,wandb", python=True)) 


    print("END")


if __name__ == "__main__":
    main()