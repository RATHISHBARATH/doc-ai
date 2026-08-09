# ============================================================
# DOC AI Common – Shared gRPC Stubs Package
# ============================================================

"""
Shared gRPC stubs for the DOC AI Ecosystem.
"""

from .inference_pb2 import InferRequest, InferResponse
from .inference_pb2_grpc import InferenceStub

__all__ = [
    "InferRequest",
    "InferResponse",
    "InferenceStub",
]