#!/bin/bash
# Development setup script for speaches

echo "Setting up speaches development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv sync --dev

# Install CLI package
echo "Installing CLI package..."
cd packages/speaches-cli
uv add --editable . --dev
cd ../..

# Create symlink for CLI (optional convenience)
echo "Creating CLI symlink..."
if [ -L "speaches" ]; then
    rm speaches
fi
ln -sf .venv/bin/speaches-cli speaches

echo "Setup complete!"
echo ""
echo "Usage:"
echo "  source .venv/bin/activate  # Activate venv"
echo "  speaches-cli --help         # Use CLI"
echo "  ./speaches --help           # Or use symlink"
echo ""
echo "To start the server:"
echo "  uv run python main.py --host 0.0.0.0 --port 9000"
echo ""
echo "Or use Docker:"
echo "  sudo docker compose -f compose.cpu.yaml up -d"