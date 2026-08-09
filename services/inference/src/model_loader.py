# ============================================================
# DOC AI Inference – Model Loader with LoRA Adapter (Corrected)
# ============================================================

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.common.minio_client import MinIOClient  # reuse from data pipeline/trainer
from src.config import config

logger = logging.getLogger(__name__)


class Model:
    """Wrapper for a Hugging Face causal language model with batch support and LoRA adapter."""

    def __init__(
        self,
        model_name: str = "distilgpt2",
        adapter_version: Optional[str] = "distilgpt2_lora_8_3epochs",
        use_gpu: str = "auto",
        quantize: str = "none",
    ):
        self.model_name = model_name
        self.adapter_version = adapter_version
        self.device = self._determine_device(use_gpu)
        self.quantize = quantize

        logger.info(f"Loading base model: {model_name} on device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token = eos_token for batch padding")

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            load_in_4bit=(quantize == "4bit"),
        )
        logger.info(f"Base model loaded with {self.model.num_parameters():,} parameters")

        # If adapter version is provided, load and apply it from MinIO
        if self.adapter_version:
            self._load_adapter_from_minio()

        self.model.eval()
        logger.info("Model ready for inference")

    def _determine_device(self, use_gpu: str) -> str:
        """Decide whether to use GPU or CPU based on config and availability."""
        if use_gpu == "false":
            return "cpu"
        if use_gpu == "true":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_adapter_from_minio(self) -> None:
        """Download the LoRA adapter from MinIO and apply it to the base model."""
        if not self.adapter_version:
            return

        minio_client = MinIOClient()
        remote_path = f"models/{self.adapter_version}/"

        # Check if the adapter exists in MinIO
        if not minio_client.object_exists(f"{remote_path}adapter_config.json"):
            logger.warning(f"Adapter {self.adapter_version} not found in MinIO. Using base model only.")
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir)
            # Download all adapter files
            for file in ["adapter_config.json", "adapter_model.safetensors", "README.md"]:
                try:
                    minio_client.download_file(f"{remote_path}{file}", local_dir / file)
                except Exception:
                    logger.debug(f"File {file} not found in adapter bundle, skipping.")
            logger.info(f"Adapter downloaded to {local_dir}")

            # Apply adapter to base model
            try:
                self.model = PeftModel.from_pretrained(self.model, str(local_dir))
                logger.info(f"LoRA adapter applied: {self.adapter_version}")
            except Exception as e:
                logger.error(f"Failed to load LoRA adapter: {e}")
                raise

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Single‑prompt generation."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.0),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_batch(self, prompts: List[str], max_tokens: int, temperature: float) -> List[str]:
        """
        Batch generation for multiple prompts.
        Returns a list of generated texts in the same order as the input prompts.
        """
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.0),
                pad_token_id=self.tokenizer.pad_token_id,
            )

        results = []
        for i, output in enumerate(outputs):
            input_len = inputs["input_ids"][i].size(0)
            generated = output[input_len:]
            results.append(self.tokenizer.decode(generated, skip_special_tokens=True))

        return results