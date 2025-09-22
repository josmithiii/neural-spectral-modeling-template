#!/usr/bin/env python3
import os
import re

def extract_info_final(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read()

        # Extract runtime (real time)
        runtime_match = re.search(r'real\s+(\d+m[\d\.]+s)', content)
        runtime = runtime_match.group(1) if runtime_match else 'N/A'

        # Extract trainable params
        params_match = re.search(r'Trainable params:\s+([\d\.]+ [KMG]?)', content)
        params = params_match.group(1) if params_match else 'N/A'

        # Look for test results - simpler approach
        test_metrics = re.findall(r'│\s+(test/[\w_]+)\s+│\s+([\d\.]+)', content)

        if test_metrics:

            # Categorize metrics
            acc_metrics = [(k, float(v)) for k, v in test_metrics if 'acc' in k]
            loss_metrics = [(k, float(v)) for k, v in test_metrics if 'loss' in k]
            mae_metrics = [(k, float(v)) for k, v in test_metrics if 'mae' in k]

            # Determine loss type and best metric
            if len(acc_metrics) > 1:
                loss_type = 'JND-weighted'
                avg_acc = sum(v for k, v in acc_metrics) / len(acc_metrics)
                best_metric = f'{avg_acc:.4f}'
            elif 'ordinal' in filename:
                loss_type = 'Ordinal'
                best_metric = f'{loss_metrics[0][1]:.4f}' if loss_metrics else 'N/A'
            elif mae_metrics:
                loss_type = 'MSE/MAE'
                best_metric = f'{mae_metrics[0][1]:.4f}'
            elif acc_metrics:
                loss_type = 'CrossEntropy'
                best_metric = f'{acc_metrics[0][1]:.4f}'
            elif loss_metrics:
                loss_type = 'CrossEntropy'
                best_metric = f'{loss_metrics[0][1]:.4f}'
            else:
                loss_type = 'Unknown'
                best_metric = 'N/A'
        else:
            # No test section found - determine why
            if 'EXPERIMENT COMPLETED SUCCESSFULLY' in content:
                loss_type = 'No test phase'
                best_metric = 'N/A'
            elif runtime != 'N/A':
                loss_type = 'Incomplete'
                best_metric = 'N/A'
            else:
                loss_type = 'Failed/Incomplete'
                best_metric = 'N/A'

        return filename, loss_type, best_metric, runtime, params

    except Exception as e:
        return filename, 'Error', str(e)[:20], 'N/A', 'N/A'

# Process all log files
os.chdir('/Users/jos/w/lightning-hydra-template-extended/experiment_logs')
log_files = [f for f in os.listdir('.') if f.endswith('-log.txt')]
results = []

for log_file in sorted(log_files):
    results.append(extract_info_final(log_file))

# Print results in structured format
print('| Experiment Name | Loss Type | Best Acc/Loss | Runtime | Parameters |')
print('|-----------------|-----------|---------------|---------|------------|')
for result in results:
    exp_name = result[0].replace('-log.txt', '')
    print(f'| {exp_name:<30} | {result[1]:<13} | {result[2]:<13} | {result[3]:<9} | {result[4]} |')