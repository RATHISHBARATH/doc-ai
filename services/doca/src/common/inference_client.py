# ============================================================
# DOC AI DOCA Service – Inference Client (Corrected)
# ============================================================

import logging
from typing import Optional

import grpc
from src.common.config import get_config, DOCAConfig

from doc_ai_common import InferRequest, InferResponse, InferenceStub
from doc_ai_common.inference_pb2 import InferRequest, InferResponse
from doc_ai_common.inference_pb2_grpc import InferenceStub

logger = logging.getLogger(__name__)


class InferenceClient:
    """
    gRPC client for the inference service.
    """

    def __init__(self, config: Optional[DOCAConfig] = None):
        if config is None:
            config = get_config()
        self.config = config
        self.addr = config.inference.grpc_addr
        self.timeout = config.inference.timeout_seconds
        self._channel = None
        self._stub = None
        self._connect()

    def _connect(self) -> None:
        try:
            self._channel = grpc.insecure_channel(self.addr)
            self._stub = InferenceStub(self._channel)
            logger.info(f"Connected to inference service at {self.addr}")
        except Exception as e:
            logger.error(f"Failed to connect to inference service: {e}")
            self._stub = None

    def infer(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
        model_name: Optional[str] = None,
    ) -> Optional[str]:
        if self._stub is None:
            logger.error("Inference stub not available")
            return None

        try:
            request = InferRequest(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                model_name=model_name or "",
            )
            response = self._stub.Infer(request, timeout=self.timeout)
            logger.debug(f"Inference succeeded for prompt: {prompt[:50]}...")
            return response.text
        except grpc.RpcError as e:
            logger.error(f"gRPC inference call failed: {e.code()} - {e.details()}")
            return None
        except Exception as e:
            logger.error(f"Unexpected inference error: {e}")
            return None

    def close(self) -> None:
        if self._channel:
            self._channel.close()
            logger.info("Inference client channel closed")