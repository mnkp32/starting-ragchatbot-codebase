#!/bin/bash

# Code Quality Check Script
# This script runs all code quality tools for the project

set -e

echo "🔍 Running Code Quality Checks..."
echo "=================================="

# Change to project root
cd "$(dirname "$0")/.."

echo "📦 Installing/updating dependencies..."
uv sync --group dev

echo ""
echo "🎨 Running Black (code formatter)..."
uv run black --check backend/ main.py

echo ""
echo "📑 Running isort (import sorter)..."
uv run isort --check-only backend/ main.py

echo ""
echo "🔍 Running flake8 (linter)..."
uv run flake8 backend/ main.py

echo ""
echo "✅ All quality checks passed!"