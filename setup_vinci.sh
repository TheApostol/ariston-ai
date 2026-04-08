#!/bin/bash

# Create folders
mkdir -p vinci_core/models
mkdir -p vinci_core/routing
mkdir -p vinci_core/workflows
mkdir -p vinci_core/context
mkdir -p vinci_core/safety

# Create __init__.py in all dirs
touch vinci_core/__init__.py
touch vinci_core/models/__init__.py
touch vinci_core/routing/__init__.py
touch vinci_core/workflows/__init__.py
touch vinci_core/context/__init__.py
touch vinci_core/safety/__init__.py

# Create core files
touch vinci_core/models/base.py
touch vinci_core/models/openai_model.py
touch vinci_core/models/anthropic_model.py

touch vinci_core/routing/model_router.py

touch vinci_core/context/builder.py

touch vinci_core/workflows/clinical.py
touch vinci_core/workflows/pharma.py
touch vinci_core/workflows/data.py

touch vinci_core/safety/guardrails.py

echo "✅ Vinci Core structure created"
