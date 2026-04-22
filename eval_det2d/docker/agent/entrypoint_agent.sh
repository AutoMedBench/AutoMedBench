#!/bin/bash
set -e

echo "[Container] Agent container starting..."
echo "[Container] Task: $TASK  Tier: $TIER  Agent: $AGENT_NAME"
echo "[Container] Patients: $PATIENT_IDS"
echo "[Container] GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (no GPU)"

# Verify isolation: these paths must NOT exist
for forbidden in /data/private /eval /results; do
    if [ -d "$forbidden" ] || [ -f "$forbidden" ]; then
        echo "FATAL: forbidden path $forbidden is accessible — aborting"
        exit 99
    fi
done

echo "[Container] Isolation check passed."

cd /workspace
exec python3 /app/agent_loop.py
