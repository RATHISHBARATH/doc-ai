#!/bin/bash
set -e

CERTS_DIR="$(dirname "$0")"
cd "$CERTS_DIR"

CA_DAYS=3650
SERVER_DAYS=365

echo "Generating mTLS certificates in $CERTS_DIR ..."

# CA
openssl genrsa -out ca.key 2048
openssl req -x509 -new -nodes -key ca.key -sha256 -days $CA_DAYS \
  -out ca.crt -subj "/CN=DOC AI CA" -addext "basicConstraints=CA:TRUE"

# Helper function to sign a certificate with SANs
sign_cert() {
  local name=$1
  local csr=$2
  local crt=$3
  local san_file=$(mktemp)
  echo "subjectAltName=DNS:$name,DNS:localhost,IP:127.0.0.1" > "$san_file"
  openssl x509 -req -in "$csr" -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out "$crt" -days $SERVER_DAYS -sha256 -extfile "$san_file"
  rm -f "$san_file"
}

# Gateway
openssl genrsa -out gateway.key 2048
openssl req -new -key gateway.key -out gateway.csr -subj "/CN=gateway" -addext "subjectAltName=DNS:gateway,DNS:localhost,IP:127.0.0.1"
sign_cert gateway gateway.csr gateway.crt

# Auth
openssl genrsa -out auth.key 2048
openssl req -new -key auth.key -out auth.csr -subj "/CN=auth" -addext "subjectAltName=DNS:auth,DNS:localhost,IP:127.0.0.1"
sign_cert auth auth.csr auth.crt

# Inference
openssl genrsa -out inference.key 2048
openssl req -new -key inference.key -out inference.csr -subj "/CN=inference" -addext "subjectAltName=DNS:inference,DNS:localhost,IP:127.0.0.1"
sign_cert inference inference.csr inference.crt

rm -f *.csr
chmod 600 *.key
chmod 644 *.crt ca.crt

echo "✅ Certificates generated successfully."