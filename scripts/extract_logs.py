#!/usr/bin/env python3
import argparse
import csv
import os
import re
from datetime import datetime

HEAD_NAMES = ("log10_decay_time", "wah_position")
METRIC_PRIORITY = ("loss", "mae", "rmse", "mse", "acc")
BATCH_SIZE_PATTERN = re.compile(r"batch_size=([0-9]+)")
MAX_EPOCHS_PATTERN = re.compile(r"max_epochs=([0-9]+)")
ACTUAL_EPOCHS_PATTERN = re.compile(r"TRAINING COMPLETED: actual_epochs=([0-9]+)")
LOSS_TYPE_PATTERN = re.compile(r"EXPERIMENT CONFIG: loss_type=([^\s]+)")


ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
TIMESTAMP_PATTERN = re.compile(r"\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),(\d{3})\]")


def _strip_ansi(text: str) -> str:
    return ANSI_PATTERN.sub('', text)


def _parse_test_metrics(content: str):
    pattern = re.compile(r"│\s*(test/[^\s│]+)\s*│\s*([-+0-9.eE]+)")
    metrics = {}
    for name, raw_value in pattern.findall(content):
        try:
            metrics[name] = float(raw_value)
        except ValueError:
            continue
    return metrics


def _select_head_metric(metrics: dict, head: str):
    prefix = f"test/{head}"
    candidates = {k: v for k, v in metrics.items() if k.startswith(prefix)}
    if not candidates:
        return None, None, None

    for suffix in METRIC_PRIORITY:
        key = f"{prefix}_{suffix}"
        if key in candidates:
            return candidates[key], suffix, _get_metric_direction(suffix)

    name, value = next(iter(candidates.items()))
    metric_label = name.split("_")[-1] if "_" in name else name.split("/")[-1]
    return value, metric_label, _get_metric_direction(metric_label)


def _get_metric_direction(metric_type: str) -> str:
    """Return arrow indicating better direction: ↑ for higher-is-better, ↓ for lower-is-better"""
    if metric_type in ('acc', 'accuracy'):
        return '↑'  # Higher accuracy is better
    elif metric_type in ('loss', 'mae', 'rmse', 'mse'):
        return '↓'  # Lower error/loss is better
    else:
        return '↑'  # Default to higher-is-better


def _format_metric(value):
    return f"{value:.4f}" if value is not None else "N/A"


def _format_runtime_from_seconds(seconds: float) -> str:
    if seconds < 0:
        return 'N/A'
    seconds = round(seconds)
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}m{remainder}s"


def _derive_runtime_from_timestamps(content: str) -> str:
    matches = TIMESTAMP_PATTERN.findall(content)
    if len(matches) < 2:
        return 'N/A'

    start_date, start_time, start_ms = matches[0]
    end_date, end_time, end_ms = matches[-1]

    try:
        start_dt = datetime.strptime(f"{start_date} {start_time}.{start_ms}", "%Y-%m-%d %H:%M:%S.%f")
        end_dt = datetime.strptime(f"{end_date} {end_time}.{end_ms}", "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return 'N/A'

    delta = (end_dt - start_dt).total_seconds()
    if delta <= 0:
        return 'N/A'
    return _format_runtime_from_seconds(delta)


def extract_info_final(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read()

        clean_content = _strip_ansi(content)

        # Extract runtime (real time)
        runtime_match = re.search(r'real\s+(\d+m[\d\.]+s)', clean_content)
        if runtime_match:
            runtime = runtime_match.group(1)
        else:
            runtime = _derive_runtime_from_timestamps(clean_content)

        # Extract trainable params
        params_match = re.search(r'Trainable params:\s+([\d\.]+ [KMG]?)', clean_content)
        params = params_match.group(1) if params_match else 'N/A'

        metrics = _parse_test_metrics(clean_content)

        batch_size_match = BATCH_SIZE_PATTERN.search(clean_content)
        batch_size = batch_size_match.group(1) if batch_size_match else 'N/A'

        max_epochs_match = MAX_EPOCHS_PATTERN.search(clean_content)
        max_epochs = max_epochs_match.group(1) if max_epochs_match else 'N/A'

        # Try to get actual epochs completed, fall back to max_epochs
        actual_epochs_match = ACTUAL_EPOCHS_PATTERN.search(clean_content)
        num_epochs = actual_epochs_match.group(1) if actual_epochs_match else max_epochs

        # Extract configured loss type from experiment config log
        loss_type_match = LOSS_TYPE_PATTERN.search(clean_content)
        if loss_type_match:
            loss_type = loss_type_match.group(1)
        else:
            # Fallback to old heuristic method for older logs
            if 'ordinal' in filename:
                loss_type = 'ordinal'
            elif 'regression' in filename:
                loss_type = 'normalized_regression'
            else:
                loss_type = 'cross_entropy'

        if metrics:
            head_values = {}
            head_directions = {}
            for head in HEAD_NAMES:
                value, metric_type, direction = _select_head_metric(metrics, head)
                head_values[head] = value
                head_directions[head] = direction

            relevant_values = [v for v in head_values.values() if v is not None]
            fallback_used = False
            if relevant_values:
                aggregate_metric = sum(relevant_values) / len(relevant_values)
                # Aggregate direction: use majority direction from heads, or ↑ for mixed
                directions = [d for d in head_directions.values() if d is not None]
                aggregate_direction = max(set(directions), key=directions.count) if directions else '↑'
            elif 'test/loss' in metrics:
                aggregate_metric = metrics['test/loss']
                aggregate_direction = '↓'  # Loss is lower-is-better
                fallback_used = True
            elif metrics:
                aggregate_metric = next(iter(metrics.values()))
                aggregate_direction = '↓'  # Default fallback assumes loss-like metric
                fallback_used = True
            else:
                aggregate_metric = None
                aggregate_direction = '↑'

            per_head_display = {head: _format_metric(head_values[head]) for head in HEAD_NAMES}
            aggregate_display = _format_metric(aggregate_metric)
            if fallback_used and aggregate_display != 'N/A':
                aggregate_display += '*'
        else:
            if 'EXPERIMENT COMPLETED SUCCESSFULLY' in content:
                loss_type = 'No test phase'
            elif runtime != 'N/A':
                loss_type = 'Incomplete'
            else:
                loss_type = 'Failed/Incomplete'
            aggregate_display = 'N/A'
            aggregate_direction = '↑'
            per_head_display = {head: 'N/A' for head in HEAD_NAMES}
            head_directions = {head: '↑' for head in HEAD_NAMES}

        return {
            'filename': filename,
            'loss_type': loss_type,
            'aggregate_metric': aggregate_display,
            'aggregate_direction': aggregate_direction,
            'runtime': runtime,
            'params': params,
            'head_metrics': per_head_display,
            'head_directions': head_directions,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
        }

    except Exception as e:
        return {
            'filename': filename,
            'loss_type': 'Error',
            'aggregate_metric': str(e)[:20],
            'aggregate_direction': '↑',
            'runtime': 'N/A',
            'params': 'N/A',
            'head_metrics': {head: 'N/A' for head in HEAD_NAMES},
            'head_directions': {head: '↑' for head in HEAD_NAMES},
            'batch_size': 'N/A',
            'num_epochs': 'N/A',
        }

# Parse command line arguments
parser = argparse.ArgumentParser(description='Extract experiment log information')
parser.add_argument('--csv', nargs='?', const='experiment_results.csv',
                    help='Write CSV output to specified file (default: experiment_results.csv)')
args = parser.parse_args()

# Process all log files
os.chdir('./experiment_logs')
log_files = [f for f in os.listdir('.') if f.endswith('-log.txt')]
results = []

for log_file in sorted(log_files):
    results.append(extract_info_final(log_file))

# Print results in structured format
header = (
    '| Experiment Name | Loss Type | Aggregate Metric | '
    'log10_decay_time | wah_position | Batch Size | Num Epochs | Runtime | Parameters |'
)
print(header)
print('|-----------------|-----------|------------------|------------------|----------------|------------|------------|---------|------------|')
for result in results:
    exp_name = result['filename'].replace('-log.txt', '')
    head_metrics = result['head_metrics']
    head_directions = result['head_directions']

    # Format metrics with arrows
    aggregate_metric = f"{result['aggregate_metric']}{result['aggregate_direction']}"
    decay_metric = f"{head_metrics['log10_decay_time']}{head_directions['log10_decay_time']}"
    wah_metric = f"{head_metrics['wah_position']}{head_directions['wah_position']}"

    print(
        f"| {exp_name:<30} | {result['loss_type']:<11} | {aggregate_metric:<16} | "
        f"{decay_metric:<16} | {wah_metric:<14} | {result['batch_size']:<10} | {result['num_epochs']:<10} | {result['runtime']:<8} | {result['params']} |"
    )

print('\nNotes:')
print('- Loss Type shows the configured loss function from model config (e.g., cross_entropy, normalized_regression, ordinal).')
print('- Classification models (cross_entropy, ordinal) use JND-weighted accuracy metrics; regression models use MSE/MAE loss functions.')
print('- Arrows indicate optimization direction: ↑ for higher-is-better (accuracies), ↓ for lower-is-better (losses/errors).')
print('- Aggregate Metric is the mean of the available per-head test metrics for log10_decay_time and wah_position (falls back to test/loss when heads are missing).')
print('- Values marked with * indicate fallback to test/loss due to missing head metrics.')
print('- Per-head columns report the exact metric logged (accuracy for classification heads, MAE for regression heads); values are rounded to 4 decimals.')
print('- Batch Size is parsed from the Hydra data configuration line.')
print('- Num Epochs shows actual epochs completed when available (from training completion log), otherwise falls back to configured max_epochs.')
print('- Runtime uses the shell `real` timer when present (falls back to log timestamps otherwise); Parameters come from the Lightning model summary output.')

# Write CSV output if requested
if args.csv:
    csv_filename = args.csv
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Experiment_Name', 'Loss_Type', 'Aggregate_Metric', 'log10_decay_time', 'wah_position', 'Batch_Size', 'Num_Epochs', 'Runtime', 'Parameters']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            exp_name = result['filename'].replace('-log.txt', '')
            head_metrics = result['head_metrics']
            writer.writerow({
                'Experiment_Name': exp_name,
                'Loss_Type': result['loss_type'],
                'Aggregate_Metric': result['aggregate_metric'],
                'log10_decay_time': head_metrics['log10_decay_time'],
                'wah_position': head_metrics['wah_position'],
                'Batch_Size': result['batch_size'],
                'Num_Epochs': result['num_epochs'],
                'Runtime': result['runtime'],
                'Parameters': result['params']
            })

    print(f'\nCSV file written: {csv_filename}')
