#!/bin/bash
set -e

PROTO_DIR=shared/proto
RUST_OUT=shared/rust/src
PYTHON_OUT=shared/python/src

# Generate Rust
cargo install protobuf-codegen
protoc --proto_path=$PROTO_DIR --rust_out=$RUST_OUT $PROTO_DIR/*.proto

# Generate Python
pip install grpcio-tools
python -m grpc_tools.protoc --proto_path=$PROTO_DIR --python_out=$PYTHON_OUT --grpc_python_out=$PYTHON_OUT $PROTO_DIR/*.proto

echo "Proto generation complete."
