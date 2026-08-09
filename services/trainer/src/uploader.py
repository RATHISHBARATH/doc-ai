# ============================================================
# DOC AI Trainer – Model Uploader & Registry
# ============================================================

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.config import TrainerConfig
from src.common.minio_client import MinIOClient
from src.common.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


def save_model_to_minio(
    local_dir: Path,
    version: str,
    client: MinIOClient,
    remote_base: Optional[str] = None,
) -> str:
    """
    Upload the trained model (or adapter) from a local directory to MinIO.

    Args:
        local_dir: Path to the directory containing the saved model/adapter.
        version: Model version string (used for folder naming).
        client: MinIO client instance.
        remote_base: Base remote path (defaults to f"models/{version}/").

    Returns:
        The remote base path where the model was uploaded.
    """
    if remote_base is None:
        remote_base = f"models/{version}/"

    # Ensure remote_base ends with a slash
    if not remote_base.endswith("/"):
        remote_base += "/"

    # Upload all files in the local directory recursively
    for file_path in local_dir.glob("**/*"):
        if file_path.is_file():
            remote_path = remote_base + file_path.relative_to(local_dir).as_posix()
            client.upload_file(file_path, remote_path)
            logger.debug(f"Uploaded {file_path} -> {remote_path}")

    logger.info(f"Model uploaded to MinIO at {remote_base}")
    return remote_base


def register_model_in_postgres(
    version: str,
    config: TrainerConfig,
    metrics: Dict[str, Any],
    postgres_client: PostgresClient,
    minio_path: str,
) -> int:
    """
    Register the trained model metadata in PostgreSQL.

    Args:
        version: Unique model version string.
        config: The trainer configuration used for this run.
        metrics: Training metrics (loss, steps, etc.).
        postgres_client: PostgreSQL client instance.
        minio_path: The remote path in MinIO where the model is stored.

    Returns:
        The ID of the inserted model record.
    """
    # Prepare metadata
    metadata = {
        "model_name": config.model_name,
        "base_model_revision": config.base_model_revision,
        "use_lora": config.use_lora,
        "lora_r": config.lora.r,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "target_modules": config.lora.target_modules,
        "bias": config.lora.bias,
        "task_type": config.lora.task_type,
        "num_epochs": config.training.num_epochs,
        "batch_size": config.training.per_device_train_batch_size,
        "learning_rate": config.training.learning_rate,
        "lr_scheduler": config.training.lr_scheduler_type,
        "use_mixed_precision": config.training.use_mixed_precision,
        "use_4bit": config.training.use_4bit,
        "max_seq_length": config.data.max_seq_length,
    }

    # We need to create a new table 'models' in PostgreSQL.
    # Since we are using the existing PostgresClient, we can execute a raw query.
    # For now, we'll use a simple INSERT statement with JSONB fields.

    # Ensure the models table exists (idempotent creation)
    with postgres_client.get_cursor(commit=True) as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id SERIAL PRIMARY KEY,
                version VARCHAR(50) NOT NULL UNIQUE,
                base_model VARCHAR(255) NOT NULL,
                adapter_type VARCHAR(50),
                lora_r INTEGER,
                lora_alpha INTEGER,
                lora_dropout FLOAT,
                num_epochs INTEGER,
                batch_size INTEGER,
                learning_rate FLOAT,
                trained_on VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                minio_path TEXT NOT NULL,
                metrics JSONB,
                metadata JSONB
            );
        """)

    # Insert the new model record
    with postgres_client.get_cursor(commit=True) as cursor:
        cursor.execute("""
            INSERT INTO models (
                version,
                base_model,
                adapter_type,
                lora_r,
                lora_alpha,
                lora_dropout,
                num_epochs,
                batch_size,
                learning_rate,
                trained_on,
                minio_path,
                metrics,
                metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (
            version,
            config.model_name,
            "lora" if config.use_lora else "full",
            config.lora.r if config.use_lora else None,
            config.lora.alpha if config.use_lora else None,
            config.lora.dropout if config.use_lora else None,
            config.training.num_epochs,
            config.training.per_device_train_batch_size,
            config.training.learning_rate,
            config.data.dataset_path,
            minio_path,
            json.dumps(metrics),
            json.dumps(metadata),
        ))
        result = cursor.fetchone()
        model_id = result[0] if result else None
        logger.info(f"Registered model in PostgreSQL with ID {model_id}")
        return model_id