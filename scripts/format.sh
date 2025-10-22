#!/bin/bash

# Code Formatting Script
# This script formats all Python code in the project

set -e

echo "🎨 Formatting Code..."
echo "===================="

# Change to project root
cd "$(dirname "$0")/.."

echo "📦 Installing/updating dependencies..."
uv sync --group dev

echo ""
echo "🎨 Running Black (code formatter)..."
uv run black backend/ main.py

echo ""
echo "📑 Running isort (import sorter)..."
uv run isort backend/ main.py

echo ""
echo "✅ Code formatting complete!"