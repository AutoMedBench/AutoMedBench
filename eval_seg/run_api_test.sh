#!/usr/bin/env bash
# ================================================================
# MedAgentsBench — API Test Runner
#
# Usage:
#   # Mock test (no real inference, no API key needed for tools):
#   bash run_api_test.sh --agent claude-opus-4-6 --task kidney --mock
#
#   # Real run:
#   export ANTHROPIC_API_KEY="sk-..."
#   bash run_api_test.sh --agent claude-opus-4-6 --task kidney
#
#   # Run all agents on all tasks:
#   bash run_api_test.sh --all --mock
# ================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Parse --all flag
RUN_ALL=false
PASSTHROUGH_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--all" ]; then
        RUN_ALL=true
    else
        PASSTHROUGH_ARGS+=("$arg")
    fi
done

echo "=========================================="
echo " MedAgentsBench — API Test"
echo "=========================================="

if [ "$RUN_ALL" = true ]; then
    AGENTS=(
        claude-opus-4-6
        claude-sonnet-4-6
        gpt-5.4
        o3-pro
        gemini-3.1-pro
        gemini-3-flash
        qwen3.5-397b
        kimi-k2.5
        deepseek-v4
        deepseek-r1
        glm-5.1
    )
    TASKS=(kidney liver)
    MOCK_FLAG=""
    for arg in "${PASSTHROUGH_ARGS[@]}"; do
        if [ "$arg" = "--mock" ]; then
            MOCK_FLAG="--mock"
        fi
    done

    for TASK in "${TASKS[@]}"; do
        for AGENT in "${AGENTS[@]}"; do
            echo ""
            echo "--- Running: $AGENT / $TASK $MOCK_FLAG ---"
            python3 agent_runner.py --agent "$AGENT" --task "$TASK" $MOCK_FLAG || \
                echo "  [WARN] $AGENT/$TASK failed — continuing..."
        done
    done

    echo ""
    echo "=========================================="
    echo " All runs complete."
    echo " Reports in: $SCRIPT_DIR/runs/"
    echo "=========================================="
else
    python3 agent_runner.py "${PASSTHROUGH_ARGS[@]}"
fi
