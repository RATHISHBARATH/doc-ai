# test_inference.py
import grpc
import inference_pb2
import inference_pb2_grpc

# Connect to the inference service directly (bypassing gateway)
channel = grpc.insecure_channel('localhost:50053')
stub = inference_pb2_grpc.InferenceStub(channel)

print("--- Test 1: Unary Inference ---")
request = inference_pb2.InferRequest(
    prompt="Hello, who are you?",
    max_tokens=20,
    temperature=0.8
)
response = stub.Infer(request)
print(f"Response: {response.text}")
print(f"Latency: {response.latency_ms}ms")

print("\n--- Test 2: Streaming Inference ---")
stream_request = inference_pb2.InferRequest(
    prompt="Tell me a short story about AI.",
    max_tokens=30,
    temperature=0.9
)
print("Tokens: ", end="", flush=True)
for resp in stub.StreamInfer(stream_request):
    print(resp.text, end=" ", flush=True)
print("\n--- Streaming test complete ---")