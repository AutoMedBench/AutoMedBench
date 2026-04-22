#!/bin/bash
set -e

echo "[Eval] Evaluation container starting..."
echo "[Eval] Task: $TASK  Agent: $AGENT_NAME  Tier: $TIER  Repeat: $REPEAT_IDX"
echo "[Eval] Patients: ${PATIENT_IDS:0:80}..."

# Defense-in-depth: confirm no network (should be --network none)
if curl -s --max-time 2 http://1.1.1.1 >/dev/null 2>&1; then
    echo "WARNING: network is accessible — eval container should run with --network none"
fi

# Verify required mounts
for d in /data/private /data/public /agent_outputs /results; do
    if [ ! -d "$d" ]; then
        echo "FATAL: mount $d missing"
        exit 1
    fi
done

cd /eval
exec python3 -u /eval/run_eval.py
