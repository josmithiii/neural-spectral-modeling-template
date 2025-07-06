# Benchmarks Directory

This directory contains benchmark reports, automation scripts, and results for systematic performance evaluation of the Lightning-Hydra template.

## 📂 Structure

```
benchmarks/
├── README.md                   # This file - directory overview
├── CIFAR_BENCHMARK_REPORT.md   # Comprehensive CIFAR benchmark documentation
├── scripts/
│   └── benchmark_cifar.py      # Automated CIFAR benchmark suite
└── results/                    # Storage for benchmark results and logs
```

## 🎯 Current Benchmarks

### CIFAR Benchmark Suite
- **Report**: [`CIFAR_BENCHMARK_REPORT.md`](./CIFAR_BENCHMARK_REPORT.md)
- **Automation**: [`scripts/benchmark_cifar.py`](./scripts/benchmark_cifar.py)
- **Coverage**: CIFAR-10, CIFAR-100, CIFAR-100 Coarse across 4 architectures
- **Expected Performance**: Literature-competitive baselines documented

## 🚀 Quick Start

### Run CIFAR Benchmarks
```bash
# Automated benchmark suite
make cbs
# or
python benchmarks/scripts/benchmark_cifar.py

# Individual benchmarks
make cb10c               # CIFAR-10 CNN
make cb10cn              # CIFAR-10 ConvNeXt
make cbs10               # All CIFAR-10 architectures
```

### View Results
```bash
# Check benchmark report
cat benchmarks/CIFAR_BENCHMARK_REPORT.md

# View training logs
ls logs/train/runs/

# Future: benchmark result artifacts
ls benchmarks/results/
```

## 📈 Adding New Benchmarks

When adding new benchmark suites:

1. **Create benchmark report**: `benchmarks/NEW_BENCHMARK_REPORT.md`
2. **Add automation script**: `benchmarks/scripts/benchmark_new.py`
3. **Store results**: `benchmarks/results/new_benchmark/`
4. **Update this README**: Document the new benchmark
5. **Add Makefile targets**: For convenient execution

## 🔗 Related Documentation

- [Configuration Guide](../README-CONFIG.md) - Complete feature documentation
- [Main README](../README.md) - Original Lightning-Hydra template
- [Experiment Configs](../configs/experiment/) - Benchmark experiment definitions

---

*This benchmarks directory enables systematic performance evaluation and comparison across datasets, architectures, and configurations in the Lightning-Hydra template.*
