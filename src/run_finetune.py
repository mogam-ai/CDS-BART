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
    compute_multi_class_classification_metrics,
    compute_regression_metrics,
    get_time,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@get_time
@hydra.main(version_base="1.1", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(watermark(packages="torch,lightning,transformers,wandb", python=True))

    # load the config file
    config = OmegaConf.create(OmegaConf.to_container(cfg))
    print(OmegaConf.to_yaml(config))

    # Load the tokenizer for mRNA
    tokenizer = BartTokenizerFast.from_pretrained(config.data.tokenizer_name_or_path)
    print(f"Tokenizer loaded from {config.data.tokenizer_name_or_path}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Tokenizer model max length: {tokenizer.model_max_length}")
    print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")

    # Load dataset or use load_from_disk for custom datasets
    dataset = load_dataset(config.data.dataset_name_or_path)
    print(dataset)

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
    print(dset)

    train_dataset = dset["train"]
    val_dataset = dset["val"]
    test_dataset = dset["test"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

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
        else compute_multi_class_classification_metrics
    )

    if config.general.wandb:
        import wandb

        # Initialize WandB
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            name=config.wandb.name,
            tags=config.wandb.tags,
            config=config,
        )
        print("WandB initialized successfully.")

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


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
