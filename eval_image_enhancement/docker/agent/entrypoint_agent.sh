#!/bin/bash
set -e

echo "[Container] Agent container starting..."
echo "[Container] Task: $TASK  Tier: $TIER  Agent: $AGENT_NAME  Repeat: $REPEAT_IDX"
echo "[Container] Patients: ${PATIENT_IDS:0:80}..."
echo "[Container] GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (no GPU)"

# Defense-in-depth: these paths must NOT exist inside the agent container.
# The orchestrator does not mount them, but verify anyway.
for forbidden in /data/private /eval /results /bands; do
    if [ -d "$forbidden" ] || [ -f "$forbidden" ]; then
        echo "FATAL: forbidden path $forbidden is accessible — aborting"
        exit 99
    fi
done

echo "[Container] Isolation check passed."

cd /workspace
exec python3 -u /app/agent_loop.py
