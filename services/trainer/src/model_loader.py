# ============================================================
# DOC AI Trainer – Model Loader (Hardcoded Target Modules)
# ============================================================

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

from src.config import TrainerConfig

logger = logging.getLogger(__name__)


def load_base_model_and_tokenizer(
    config: TrainerConfig,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load the base model and tokenizer from Hugging Face Hub (or local cache).

    This is a temporary dependency on `transformers`; in the future, this
    will be replaced with a DOC‑native model loader.
    """
    model_name = config.model_name
    logger.info(f"Loading base model: {model_name}")

    # Load tokenizer (if not provided)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=config.trust_remote_code,
            use_fast=config.use_fast_tokenizer,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Determine torch dtype and device
    if torch.cuda.is_available():
        device_map = "auto"
        torch_dtype = torch.float16 if config.training.use_mixed_precision == "fp16" else torch.bfloat16
    else:
        device_map = None
        torch_dtype = torch.float32

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_map,
        use_cache=True,
    )

    logger.info(f"Base model loaded with {model.num_parameters():,} parameters")
    return model, tokenizer


def apply_lora(
    model: PreTrainedModel,
    config: TrainerConfig,
) -> PreTrainedModel:
    """
    Apply LoRA adapters to the model using the `peft` library.

    This is a temporary dependency; a DOC‑native LoRA implementation will
    replace it in the future.
    """
    if not config.use_lora:
        logger.info("LoRA is disabled; using full fine‑tuning.")
        return model

    # ---- HARDCODED OVERRIDE FOR distilgpt2 ----
    # This ensures the correct target modules are used even if training.yaml is incorrect.
    if config.model_name == "distilgpt2":
        target_modules = ["c_attn", "c_proj"]
        logger.info("Using hardcoded target modules for distilgpt2: c_attn, c_proj")
    else:
        target_modules = config.lora.target_modules
    # ------------------------------------------

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        target_modules=target_modules,   # <-- Use the hardcoded/fallback list
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    logger.info(f"Applying LoRA with r={config.lora.r}, alpha={config.lora.alpha}")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_model_for_training(
    config: TrainerConfig,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Full pipeline: load base model, optionally apply LoRA, and return model + tokenizer.

    Returns:
        model: The model ready for training (with LoRA if enabled).
        tokenizer: The corresponding tokenizer.
    """
    model, tokenizer = load_base_model_and_tokenizer(config, tokenizer)
    if config.use_lora:
        model = apply_lora(model, config)
    return model, tokenizer


def load_lora_adapter(
    model: PreTrainedModel,
    adapter_path: str,
) -> PreTrainedModel:
    """
    Load a previously trained LoRA adapter from disk (or MinIO) onto a base model.

    This is a temporary dependency; later, DOC‑native loading will be used.
    """
    logger.info(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    return model