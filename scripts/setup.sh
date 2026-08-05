#!/bin/bash
set -e  # Exit on error

echo "📦 Syncing uv dependencies..."
uv sync --all-groups

echo "📦 Setting up Node.js environment..."
uv run nodeenv --node=22.1.0 -p

echo "📦 Installing AWS CDK..."
# The CDK CLI is versioned independently of aws-cdk-lib (decoupled at v2.1000.0)
# and reads any cloud assembly schema at or below its own. aws-cdk-lib is
# unpinned in pyproject.toml, so this must stay recent enough for whatever uv
# resolves: an older CLI fails synth with "Cloud assembly schema version
# mismatch" rather than anything pointing at the real cause.
uv run npm install -g aws-cdk@2.1135.0

echo "✅ Setup complete!"
