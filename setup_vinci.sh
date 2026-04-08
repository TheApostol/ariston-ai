#!/bin/bash

# Ariston AI - Production Workspace Sync
# This script ensures the directory structure and core manifest match the finalized Phase 6 architecture.

set -e

echo "🏗️  Starting Ariston AI Workspace Sync..."

# Ensure core directories
mkdir -p vinci_core/agent
mkdir -p vinci_core/database
mkdir -p vinci_core/models
mkdir -p vinci_core/routing
mkdir -p vinci_core/workflows
mkdir -p vinci_core/safety
mkdir -p vinci_core/evaluation
mkdir -p vinci_core/middleware

# Initialize packages
find vinci_core -type d -exec touch {}/__init__.py \;

echo "📦 Ensuring core models..."
touch vinci_core/models/base_model.py
touch vinci_core/models/openai_model.py
touch vinci_core/models/anthropic_model.py
touch vinci_core/models/gemini_model.py
touch vinci_core/models/ollama_model.py
touch vinci_core/models/openrouter_model.py

echo "🧠 Ensuring routing layer..."
touch vinci_core/routing/model_router.py
touch vinci_core/routing/consensus_router.py

echo "🛡️  Ensuring safety & eval..."
touch vinci_core/safety/guardrails.py
touch vinci_core/evaluation/benchmark_logger.py

echo "✅ Workspace sync complete. Run 'pytest' to verify integrity."
