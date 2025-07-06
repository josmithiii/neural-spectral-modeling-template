#!/usr/bin/env python3
"""
CIFAR Benchmark Script

Runs systematic benchmarks on CIFAR-10 and CIFAR-100 datasets using different architectures.
Collects performance metrics for comparison and validation.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def run_experiment(experiment_name: str, epochs: int = 10, accelerator: str = "cpu") -> Tuple[str, float, float]:
    """Run a single experiment and extract final validation accuracy.

    Args:
        experiment_name: Name of the experiment configuration
        epochs: Number of epochs to train
        accelerator: Hardware accelerator to use

    Returns:
        Tuple of (experiment_name, final_val_acc, final_test_acc)
    """
    print(f"\n🚀 Running {experiment_name} for {epochs} epochs...")

    cmd = [
        "python", "src/train.py",
        f"experiment={experiment_name}",
        f"trainer.max_epochs={epochs}",
        f"trainer.accelerator={accelerator}",
        "data.num_workers=0",  # Avoid multiprocessing issues
    ]

    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        duration = time.time() - start_time

        if result.returncode != 0:
            print(f"❌ {experiment_name} failed:")
            print(result.stderr[-500:])  # Last 500 chars of error
            return experiment_name, 0.0, 0.0

        # Extract validation and test accuracy from output
        output = result.stdout
        val_acc = 0.0
        test_acc = 0.0

        # Look for validation accuracy in output
        for line in output.split('\n'):
            if 'val/acc' in line and 'val/acc_best' not in line:
                try:
                    val_acc = float(line.split('val/acc:')[1].split()[0])
                except:
                    pass
            if 'test/acc' in line:
                try:
                    test_acc = float(line.split('test/acc')[1].split()[0])
                    # Handle different formats
                    if test_acc > 1:  # Percentage format
                        test_acc /= 100
                except:
                    pass

        print(f"✅ {experiment_name}: Val Acc: {val_acc:.3f}, Test Acc: {test_acc:.3f} ({duration:.1f}s)")
        return experiment_name, val_acc, test_acc

    except subprocess.TimeoutExpired:
        print(f"⏰ {experiment_name} timed out after 5 minutes")
        return experiment_name, 0.0, 0.0
    except Exception as e:
        print(f"❌ {experiment_name} error: {e}")
        return experiment_name, 0.0, 0.0


def main():
    """Run CIFAR benchmarks and generate report."""
    print("🎯 CIFAR Benchmark Suite")
    print("=" * 50)

    # Define benchmark experiments (quick subset for demonstration)
    benchmark_configs = [
        # Quick CIFAR-10 benchmarks
        ("cifar10_benchmark_cnn", 5),
        ("cifar10_benchmark_convnext", 5),

        # Quick CIFAR-100 comparison
        ("cifar100_coarse_cnn", 5),
    ]

    results: List[Tuple[str, float, float]] = []

    print(f"Running {len(benchmark_configs)} benchmark experiments...")

    # Run all experiments
    for exp_name, epochs in benchmark_configs:
        result = run_experiment(exp_name, epochs)
        results.append(result)
        time.sleep(2)  # Brief pause between experiments

    # Generate report
    print("\n" + "=" * 80)
    print("📊 CIFAR BENCHMARK RESULTS")
    print("=" * 80)

    print(f"{'Experiment':<35} {'Val Acc':<10} {'Test Acc':<10} {'Dataset':<12} {'Architecture'}")
    print("-" * 80)

    for exp_name, val_acc, test_acc in results:
        # Parse experiment details
        if 'cifar10' in exp_name:
            dataset = "CIFAR-10"
        elif 'cifar100_coarse' in exp_name:
            dataset = "CIFAR-100 (20)"
        elif 'cifar100' in exp_name:
            dataset = "CIFAR-100 (100)"
        else:
            dataset = "Unknown"

        if 'cnn' in exp_name:
            arch = "SimpleCNN"
        elif 'convnext' in exp_name:
            arch = "ConvNeXt"
        elif 'vit' in exp_name:
            arch = "ViT"
        elif 'efficientnet' in exp_name:
            arch = "EfficientNet"
        else:
            arch = "Unknown"

        print(f"{exp_name:<35} {val_acc:.3f}     {test_acc:.3f}     {dataset:<12} {arch}")

    # Performance analysis
    print("\n" + "=" * 80)
    print("📈 PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Group by dataset
    cifar10_results = [(name, val, test) for name, val, test in results if 'cifar10' in name]
    cifar100_results = [(name, val, test) for name, val, test in results if 'cifar100' in name and 'coarse' not in name]
    cifar100_coarse_results = [(name, val, test) for name, val, test in results if 'cifar100_coarse' in name]

    if cifar10_results:
        best_cifar10 = max(cifar10_results, key=lambda x: x[1])
        print(f"🥇 Best CIFAR-10: {best_cifar10[0]} with {best_cifar10[1]:.3f} val acc")

    if cifar100_results:
        best_cifar100 = max(cifar100_results, key=lambda x: x[1])
        print(f"🥇 Best CIFAR-100: {best_cifar100[0]} with {best_cifar100[1]:.3f} val acc")

    if cifar100_coarse_results:
        best_coarse = max(cifar100_coarse_results, key=lambda x: x[1])
        print(f"🥇 Best CIFAR-100 Coarse: {best_coarse[0]} with {best_coarse[1]:.3f} val acc")

    # Expected performance ranges
    print("\n📋 EXPECTED PERFORMANCE RANGES:")
    print("CIFAR-10: 85-95% (literature baselines)")
    print("CIFAR-100: 55-75% (literature baselines)")
    print("CIFAR-100 Coarse: 75-85% (easier 20-class task)")

    print("\n✅ Benchmark complete! Check results above.")
    print("💡 For full training, run experiments with 50+ epochs for CIFAR-10, 100+ for CIFAR-100")


if __name__ == "__main__":
    main()
