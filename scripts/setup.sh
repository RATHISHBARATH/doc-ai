#!/bin/bash
set -e

echo "Setting up DOC AI development environment..."

# Check if Rust is already installed (it is pre-installed in our dev container)
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
else
    echo "Rust is already installed. Skipping installation."
fi

# Source the correct Cargo environment (dynamically find it)
if [ -n "$CARGO_HOME" ] && [ -f "$CARGO_HOME/env" ]; then
    source "$CARGO_HOME/env"
elif [ -f "/usr/local/cargo/env" ]; then
    source "/usr/local/cargo/env"
elif [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# Install Python tools (using --user avoids permission errors in the container)
pip install --user poetry pre-commit

# Install Node.js (for future)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Docker Compose
sudo apt-get install -y docker-compose

# Install pre-commit hooks
pre-commit install

echo "Setup complete. Run 'make build' to build the services."