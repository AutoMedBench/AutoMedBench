#!/usr/bin/env bash
# End-to-end dummy test for the segmentation evaluation pipeline.
# Tests kidney and liver tasks with 3 dummy agents each.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo " MedAgentsBench Segmentation - Dummy Test"
echo " Tasks: kidney, liver"
echo "=========================================="

# Step 1: Stage data
echo ""
echo "[Step 1] Staging CruzBench patient data..."
python3 stage_data.py

# Step 2: Generate dummy agent submissions
echo ""
echo "[Step 2] Generating dummy agent outputs..."
python3 make_dummy_agents.py

# Step 3: Run evaluation for each (agent x task)
echo ""
echo "[Step 3] Running evaluations..."

PATIENTS="BDMAP_00000001,BDMAP_00000005,BDMAP_00000012,BDMAP_00000017,BDMAP_00000040,BDMAP_00000080,BDMAP_00000449,BDMAP_00000862"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

for TASK in kidney liver; do
    GT_DIR="$SCRIPT_DIR/data/$TASK/private/masks"
    GT_CSV="$SCRIPT_DIR/data/$TASK/private/ground_truth.csv"
    PUB_DIR="$SCRIPT_DIR/data/$TASK/public"

    for AGENT in perfect partial empty; do
        AGENT_DIR="$SCRIPT_DIR/dummy_agents/$AGENT/$TASK/agents_outputs"

        python3 run_eval.py \
            --gt-dir "$GT_DIR" \
            --gt-csv "$GT_CSV" \
            --agent-dir "$AGENT_DIR" \
            --public-dir "$PUB_DIR" \
            --patients "$PATIENTS" \
            --task "$TASK" \
            --name "$AGENT" \
            --output-json "$RESULTS_DIR/${TASK}_${AGENT}_report.json"
    done
done

# Summary comparison
echo ""
echo "=========================================="
echo " Side-by-side Comparison"
echo "=========================================="
python3 -c "
import json, os

results_dir = '$RESULTS_DIR'
tasks = ['kidney', 'liver']
agents = ['perfect', 'partial', 'empty']

for task in tasks:
    print(f'\n--- {task.upper()} ---')
    header = f\"{'Metric':<25} {'perfect':>10} {'partial':>10} {'empty':>10}\"
    print(header)
    print('-' * len(header))

    reports = {}
    for a in agents:
        with open(os.path.join(results_dir, f'{task}_{a}_report.json')) as f:
            reports[a] = json.load(f)

    rows = [
        ('** Overall Score **', lambda r: r['aggregate']['overall_score']),
        ('Rating', lambda r: r['aggregate']['rating']),
        ('Resolved', lambda r: r['aggregate']['resolved']),
        ('Workflow Score', lambda r: r['aggregate']['agentic_score']),
        ('Clinical Score', lambda r: r['aggregate']['clinical_score']),
        ('', lambda r: ''),
        ('Organ Dice', lambda r: r['metrics'].get('organ_dice', 'N/A')),
        ('Lesion Dice', lambda r: r['metrics'].get('tumor_dice', 0)),
        ('Medal Tier', lambda r: r['metrics'].get('medal_name', '?')),
        ('', lambda r: ''),
        ('S4', lambda r: r['step_scores']['s4']),
        ('S5', lambda r: r['step_scores']['s5']),
        ('Progress Rate', lambda r: r['aggregate']['progress_rate']),
        ('Failure', lambda r: r['failure']['primary_failure'] if r.get('failure') else 'None'),
    ]
    for name, fn in rows:
        vals = []
        for a in agents:
            v = fn(reports[a])
            if v is None:
                vals.append('N/A')
            elif isinstance(v, float):
                vals.append(f'{v:.4f}')
            else:
                vals.append(str(v))
        print(f'{name:<25} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}')

print()
print(f'Done. JSON reports saved to: $RESULTS_DIR/')
"
