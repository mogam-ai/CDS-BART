import os
from functools import partial

import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import (
    BartForSequenceClassification,
    BartTokenizerFast,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from watermark import watermark

from utils import (
    apply_scaling,
    compute_classification_metrics,
    compute_regression_metrics,
    get_time,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


@get_time
@hydra.main(version_base="1.1", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(watermark(packages="torch,lightning,transformers,wandb", python=True))

    # load the config file
    config = OmegaConf.create(OmegaConf.to_container(cfg))

    # Load the tokenizer for mRNA
    tokenizer = BartTokenizerFast.from_pretrained(config.data.tokenizer_name_or_path)

    # Load dataset or use load_from_disk for custom datasets
    dataset = load_dataset(config.data.dataset_name_or_path)

    if config.data.get("scaling_method", None) is not None:
        # Apply scaling to the dataset
        print("Applying scaling to the dataset...")
        dataset = apply_scaling(
            dataset,
            scaling_method=config.data.scaling_method,
        )
        print("Dataset scaling applied successfully.")

    else:
        print("No scaling method specified, using the dataset as is.")
        dataset = dataset

    def prepocess(example, label_name):
        seq = example["seq"]
        labels = example[label_name]

        inputs = tokenizer(
            seq,
            truncation=True,
            max_length=config.data.max_len,
            padding=False,
            add_special_tokens=True,
        )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        }

    # Preprocess the dataset

    dset = dataset.map(
        partial(prepocess, label_name=config.data.label_name), batched=True, num_proc=10
    )
    print("Preprocessed dataset successfully.")

    train_dataset = dset["train"]
    val_dataset = dset["val"]
    test_dataset = dset["test"]

    # a default collator for padding
    data_collator = DataCollatorWithPadding(tokenizer)

    # Load the model for finetuning
    predictor = BartForSequenceClassification.from_pretrained(
        config.data.model_name_or_path, num_labels=config.general.num_labels
    )

    # Initialize the finetuning process
    training_args = TrainingArguments(**config.hg.training)

    compute_metrics_fn = (
        compute_regression_metrics
        if config.general.task_type == "regression"
        else compute_classification_metrics
    )

    if config.general.wandb:
        import wandb

        # Initialize WandB
        wandb.init(
            entity=config.wandb_config.entity,
            project=config.wandb_config.project,
            name=config.wandb_config.name,
            tags=config.wandb_config.tags,
            config=config,
        )
        print("WandB initialized successfully.")
        training_args.report_to = "wandb"

    else:
        print("WandB is not enabled, using default logging.")
        training_args.report_to = "none"

    # Trainer
    trainer = Trainer(
        model=predictor,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)], if you want early stop
    )

    trainer.train()
    _, _, metrics = trainer.predict(
        test_dataset=test_dataset,
        metric_key_prefix="test",
    )
    trainer.log(metrics)

    print("Training completed successfully.")
    if config.general.wandb:
        wandb.finish()
        print("WandB finished successfully.")

    print("END")


if __name__ == "__main__":
    main()
