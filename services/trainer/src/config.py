# ============================================================
# DOC AI Trainer – Configuration Loader
# ============================================================

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class LoraConfig:
    """Configuration for LoRA fine‑tuning."""
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Core training hyperparameters."""
    output_dir: str = "/app/cache/model_output"
    overwrite_output_dir: bool = True
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 0
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    logging_steps: int = 5
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False
    use_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_mixed_precision: str = "fp16"  # "fp16", "bf16", or "none"


@dataclass
class DataConfig:
    """Dataset and tokenizer configuration."""
    dataset_path: str = "final/dataset.parquet"
    tokenizer_path: str = "tokenizer/v1000_1_*/tokenizer.json"
    max_seq_length: int = 512
    dataset_split: str = "train"
    num_proc: int = 2
    shuffle_dataset: bool = True


@dataclass
class ScientistRetrievalConfig:
    """Configuration for the scientist knowledge graph."""
    use_scientist_retrieval: bool = False
    embedding_dim: int = 384
    top_k: int = 3
    vector_store_path: Optional[str] = None


@dataclass
class TrainerConfig:
    """Master configuration for the trainer service."""
    model_name: str = "distilgpt2"
    base_model_revision: str = "main"
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    use_lora: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    scientist: ScientistRetrievalConfig = field(default_factory=ScientistRetrievalConfig)
    log_level: str = "INFO"
    report_to: str = "none"
    run_name: str = "doc_ai_lora_run"

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainerConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Override with environment variables (DOC_* prefix)
        data = cls._apply_env_overrides(data)

        return cls(
            model_name=data.get("model_name", "distilgpt2"),
            base_model_revision=data.get("base_model_revision", "main"),
            trust_remote_code=data.get("trust_remote_code", False),
            use_fast_tokenizer=data.get("use_fast_tokenizer", True),
            use_lora=data.get("use_lora", True),
            lora=LoraConfig(**data.get("lora", {})),
            training=TrainingConfig(**data.get("training", {})),
            data=DataConfig(**data.get("data", {})),
            scientist=ScientistRetrievalConfig(**data.get("scientist", {})),
            log_level=data.get("log_level", "INFO"),
            report_to=data.get("report_to", "none"),
            run_name=data.get("run_name", "doc_ai_lora_run"),
        )

    @staticmethod
    def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
        """Override configuration with environment variables (DOC_*)."""
        # Simple implementation: flatten with double underscores for nesting.
        # Example: DOC_TRAINING__LEARNING_RATE sets training.learning_rate.
        for key, value in os.environ.items():
            if key.startswith("DOC_"):
                # Remove prefix and split by '__'
                parts = key[4:].lower().split("__")
                target = data
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                # Convert value to appropriate type (string, int, float, bool)
                if value.lower() in ("true", "false"):
                    target[parts[-1]] = value.lower() == "true"
                elif value.isdigit():
                    target[parts[-1]] = int(value)
                else:
                    try:
                        target[parts[-1]] = float(value)
                    except ValueError:
                        target[parts[-1]] = value
        return data


# Singleton global config
_config: Optional[TrainerConfig] = None


def load_config(config_path: Optional[Path] = None) -> TrainerConfig:
    """Load the trainer configuration from a YAML file."""
    if config_path is None:
        # Default path relative to the service root
        config_path = Path("/app/configs/training.yaml")
    return TrainerConfig.from_yaml(config_path)


def get_config(config_path: Optional[Path] = None) -> TrainerConfig:
    """Get the global config instance (loads on first call)."""
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config


def reset_config():
    """Reset the singleton (useful for testing)."""
    global _config
    _config = None