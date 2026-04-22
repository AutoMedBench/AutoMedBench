#!/bin/bash
set -e

echo "[Eval] Evaluation container starting..."
echo "[Eval] Task: $TASK  Agent: $AGENT_NAME"
echo "[Eval] Patients: $PATIENT_IDS"

# Verify we have no network (defense-in-depth)
if curl -s --max-time 2 http://1.1.1.1 >/dev/null 2>&1; then
    echo "WARNING: Network is accessible — eval container should run with --network none"
fi

# Verify ground truth exists
if [ ! -d "/data/private/masks" ]; then
    echo "FATAL: /data/private/masks not mounted"
    exit 1
fi

# Verify agent outputs exist
if [ ! -d "/agent_outputs" ]; then
    echo "FATAL: /agent_outputs not mounted"
    exit 1
fi

cd /eval

python3 -c "
import sys, json, os
sys.path.insert(0, '/eval')

from format_checker import check_submission
from dice_scorer import score_all
from medal_tier import assign_tier
from aggregate import build_report
from failure_classifier import classify_failure

patient_ids = os.environ['PATIENT_IDS'].split(',')
task = os.environ.get('TASK', 'unknown')
agent_name = os.environ.get('AGENT_NAME', 'unknown')

# Step 1: Format check
format_result = check_submission(
    agent_dir='/agent_outputs',
    patient_ids=patient_ids,
    public_dir='/data/public',
)

# Step 2: Dice scoring
dice_result = score_all(
    pred_dir='/agent_outputs',
    gt_dir='/data/private/masks',
    patient_ids=patient_ids,
)

# Step 3: Medal tier
lesion_dice = dice_result.get('mean_lesion_dice', 0.0)
medal_result = assign_tier(lesion_dice)

# Step 4: Aggregate
report = build_report(format_result, dice_result, medal_result)

# Step 5: Failure classification
failure = classify_failure(report)
report['failure'] = failure
report['_dice_per_patient'] = dice_result.get('per_patient', {})

# Write result
output_path = '/results/detail_report.json'
with open(output_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f'[Eval] Report written to {output_path}')
rating = report.get('aggregate', {}).get('rating', '?')
resolved = report.get('aggregate', {}).get('resolved', False)
overall = report.get('aggregate', {}).get('overall_score', 0)
print(f'[Eval] Rating: {rating}  Resolved: {resolved}  Overall: {overall:.4f}')
"

echo "[Eval] Done."
