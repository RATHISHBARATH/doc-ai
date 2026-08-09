# ============================================================
# DOC AI Trainer – Main Training Loop
# ============================================================

import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_scheduler,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from tqdm import tqdm
from accelerate import Accelerator

from src.config import TrainerConfig, get_config
from src.data_loader import create_dataloader, load_tokenizer_from_minio
from src.model_loader import load_model_for_training
from src.uploader import save_model_to_minio, register_model_in_postgres
from src.common.minio_client import MinIOClient  # Reuse from data pipeline
from src.common.postgres_client import PostgresClient  # Reuse from data pipeline

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class that orchestrates model fine‑tuning (LoRA) or pre‑training.
    """

    def __init__(
        self,
        config: TrainerConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataloader: DataLoader,
        accelerator: Accelerator,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.accelerator = accelerator
        self.training_config = config.training

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            betas=(self.training_config.adam_beta1, self.training_config.adam_beta2),
            eps=self.training_config.adam_epsilon,
        )

        # Learning rate scheduler
        num_training_steps = len(train_dataloader) * self.training_config.num_epochs
        self.lr_scheduler = get_scheduler(
            self.training_config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Prepare with accelerator
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
        )

    def train(self) -> dict:
        """Run the full training loop and return metrics."""
        logger.info("Starting training loop")
        self.model.train()
        global_step = 0
        total_loss = 0.0

        progress_bar = tqdm(
            range(self.training_config.num_epochs * len(self.train_dataloader)),
            desc="Training",
            disable=not logger.isEnabledFor(logging.INFO),
        )

        for epoch in range(self.training_config.num_epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    # Prepare input_ids and attention_mask
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]

                    # For causal LM, labels are the same as input_ids (shifted internally)
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    )
                    loss = outputs.loss

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.training_config.max_grad_norm,
                        )

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                global_step += 1
                epoch_loss += loss.item()
                total_loss += loss.item()

                # Logging
                if global_step % self.training_config.logging_steps == 0:
                    logger.info(
                        f"Step {global_step} – Loss: {loss.item():.4f}, LR: {self.lr_scheduler.get_last_lr()[0]:.6f}"
                    )
                progress_bar.update(1)

            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            logger.info(f"Epoch {epoch+1}/{self.training_config.num_epochs} completed. Avg loss: {avg_epoch_loss:.4f}")

        # Final metrics
        avg_loss = total_loss / global_step
        logger.info(f"Training completed. Average loss: {avg_loss:.4f}")
        return {"loss": avg_loss, "steps": global_step}


def main():
    """Entry point for the trainer service."""
    # Load configuration
    config_path = os.environ.get("DOC_CONFIG_PATH", "/app/configs/training.yaml")
    config = get_config(Path(config_path))

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize clients
    minio_client = MinIOClient()
    postgres_client = PostgresClient()

    # Load tokenizer and dataset
    tokenizer = load_tokenizer_from_minio(config, minio_client)
    train_dataloader = create_dataloader(
        config=config,
        tokenizer=tokenizer,
        client=minio_client,
        split=config.data.dataset_split,
        batch_size=config.training.per_device_train_batch_size,
        shuffle=config.data.shuffle_dataset,
    )

    # Load model (with optional LoRA)
    model, tokenizer = load_model_for_training(config, tokenizer)

    # Initialize accelerator (handles mixed precision, device placement, etc.)
    accelerator = Accelerator(
        mixed_precision=(
            config.training.use_mixed_precision
            if config.training.use_mixed_precision != "none"
            else "no"
        ),
        log_with="tensorboard" if config.report_to == "tensorboard" else None,
    )

    # Create trainer instance
    trainer = Trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        accelerator=accelerator,
    )

    # Train
    metrics = trainer.train()

    # Save model locally
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap model and save
    unwrapped_model = accelerator.unwrap_model(trainer.model)
    if isinstance(unwrapped_model, PreTrainedModel):
        unwrapped_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        # For LoRA, we can save the adapter
        unwrapped_model.save_pretrained(output_dir)

    logger.info(f"Model saved locally to {output_dir}")

    # Upload to MinIO and register in PostgreSQL
    version = f"{config.model_name}_lora_{config.lora.r}_{config.training.num_epochs}epochs"
    save_model_to_minio(output_dir, version, minio_client)
    register_model_in_postgres(
        version=version,
        config=config,
        metrics=metrics,
        postgres_client=postgres_client,
        minio_path=f"models/{version}/",
    )

    logger.info("Trainer finished successfully.")


if __name__ == "__main__":
    main()