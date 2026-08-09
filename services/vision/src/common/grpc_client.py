# ============================================================
# DOC AI Vision Service – Generic gRPC Client
# ============================================================

import grpc
import logging
from typing import Optional, Type, TypeVar

T = TypeVar('T')


class GRPCClient:
    """
    A generic gRPC client that can connect to a gRPC server and provide
    a stub for any service interface. This is used for inter‑service
    communication (e.g., Vision → DOCA, Vision → Inference).
    """

    def __init__(self, addr: str, insecure: bool = True):
        """
        Initialize the client with a target address.

        Args:
            addr: The gRPC server address (e.g., 'doca:50054' or 'inference:50053').
            insecure: If True, use an insecure channel (default). For production,
                      set to False and provide TLS credentials.
        """
        self.addr = addr
        self.insecure = insecure
        self.channel: Optional[grpc.Channel] = None
        self.logger = logging.getLogger(__name__)

    def connect(self) -> grpc.Channel:
        """
        Establish the gRPC channel. Returns the channel for further use.
        """
        if self.channel is not None:
            return self.channel

        if self.insecure:
            self.channel = grpc.insecure_channel(self.addr)
        else:
            # Placeholder for secure (mTLS) channel – to be implemented later.
            raise NotImplementedError("Secure gRPC channel not yet implemented")

        self.logger.info(f"Connected to gRPC server at {self.addr}")
        return self.channel

    def get_stub(self, stub_class: Type[T]) -> T:
        """
        Get a stub for the given service class.

        Example:
            from doc_ai_common.inference_pb2_grpc import InferenceStub
            client = GRPCClient('inference:50053')
            stub = client.get_stub(InferenceStub)
            response = stub.Infer(request)

        Args:
            stub_class: The gRPC stub class (e.g., InferenceStub).

        Returns:
            An instance of the stub bound to the channel.
        """
        if self.channel is None:
            self.connect()
        return stub_class(self.channel)

    def close(self) -> None:
        """Close the gRPC channel if it is open."""
        if self.channel is not None:
            self.channel.close()
            self.channel = None
            self.logger.info("gRPC channel closed")