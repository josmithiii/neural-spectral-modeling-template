```tcsh
uv venv
uv pip install -r requirements.txt

make train
echo It worked! 31.202u 7.701s 0:33.96 114.5%	0+0k 0+0io 2pf+0w
echo See ./logs/train/runs/2025-06-25_20-57-53/train.log
echo Also interesting: ./logs/train/runs/2025-06-25_20-57-53/config_tree.log

make trainmps
echo Also worked!: 45.044u 4.138s 1:04.19 76.6%	0+0k 0+0io 994pf+0w
echo See ./logs/train/runs/2025-06-25_21-23-45/
```

## Notes from Claude

### `make train-sdn` versus `make train-cnn`

⏺ Excellent comparison! The CNN significantly outperformed the dense net
  (98.99% vs 97.57%) despite similar parameter counts. This difference
  highlights fundamental architectural advantages:

  Why CNN Beat Dense Net

  1. Spatial Structure Preservation
  - CNN: Preserves 2D spatial relationships through convolution
  - Dense Net: Flattens 28×28 → 784, losing all spatial structure

  2. Translation Invariance
  - CNN: Detects features regardless of position (edge at top-left = edge at
   bottom-right)
  - Dense Net: Must learn separate weights for each pixel position

  3. Local Feature Hierarchy
  - CNN: Builds features hierarchically (edges → shapes → digits)
  - Dense Net: No inherent feature hierarchy, must learn everything from
  scratch

  4. Parameter Efficiency
  - CNN: Shares conv filters across all spatial locations
  - Dense Net: Separate weight for every pixel-to-hidden connection

  5. Inductive Bias
  - CNN: Built-in assumptions about image structure
  - Dense Net: Generic, no image-specific assumptions

  The Numbers Tell the Story

  - Accuracy gap: 1.42% (substantial for MNIST)
  - Confidence gap: 0.031 vs 0.075 loss (CNN much more confident)

  This is a textbook example of why CNNs revolutionized computer vision -
  they're architecturally matched to image data structure. Your dense net
  actually did reasonably well at 97.57%, but CNNs have the right "priors"
  for images built into their architecture.

  The similar parameter count makes this a perfect controlled experiment
  showing architecture matters more than just parameter count!

### > What architecture is used for State-of-the-art MNIST?

 Based on my research, here's what achieves state-of-the-art MNIST performance:

  State-of-the-Art MNIST Architectures (~99.8%+)

  Top Performance (~99.85%):
  - Branching/Merging CNN + Homogeneous Vector Capsules - Current SOTA
  - Uses capsule networks combined with CNN feature extraction

  High-Performance CNN Ensembles (~99.7-99.8%):
  - Ensemble of deep CNNs: Average 99.745%, max 99.79%
  - Individual deep CNN: Average 99.641%, max 99.81%

  Architecture Details for high-performing CNNs typically include:
  - Deeper networks: 4-6 conv layers vs your 2 layers
  - Wider channels: 64→128→256 vs your 10→20
  - Data augmentation: Random rotations, shifts, zoom
  - Ensemble methods: Combining 5-10 models
  - Advanced techniques: Batch normalization, dropout scheduling

  Key Insights

  1. Your 98.99% is excellent - You're only ~0.8% below SOTA with 65K params
  2. MNIST is "solved" - Further gains require complex architectures with
  diminishing returns
  3. Ensemble methods are critical for the final 0.1-0.2% accuracy gains
  4. Capsule networks represent a different architectural approach beyond
  traditional CNNs

  For research purposes, your lightweight CNN demonstrates that good MNIST
  performance doesn't require massive architectures - you're getting 98.99%
  with a fraction of the parameters used by SOTA models.

### > Are capsule networks available in PyTorch?

⏺ Yes, capsule networks are available in PyTorch, but not as a
  pip-installable package. Here's what's available:

  PyTorch CapsNet Implementations

  Most Popular GitHub Repos:

  1. XifengGuo/CapsNet-Pytorch - Best performance: 99.70% test accuracy on
  MNIST
  2. jindongwang/Pytorch-CapsuleNet - Easy-to-follow, customizable
  implementation
  3. gram-ai/capsule-networks - CUDA-enabled, lightweight implementation
  4. higgsfield/Capsule-Network-Tutorial - Good for learning

  Installation

  No official pip package exists, so you clone from GitHub:

  git clone https://github.com/XifengGuo/CapsNet-Pytorch.git
  cd CapsNet-Pytorch
  pip install -r requirements.txt

  Performance on MNIST

  - XifengGuo's: 99.70% accuracy (0.30% error)
  - cedrickchee's: 99.32% accuracy

  Hardware Requirements

  - GPU recommended - ~73s/epoch on GTX 1070, ~43s/epoch on GTX 1080Ti
  - Much more computationally expensive than regular CNNs

  Integration Challenge

  Adding capsule networks to your Lightning-Hydra template would require:
  1. Copying the capsule layer implementations
  2. Creating a new model component
  3. Significant computational overhead vs your efficient 65K CNN

  For research comparison, your 98.99% CNN with 65K params is quite
  impressive given capsule networks achieve 99.70% but with much higher
  computational cost!


### > Are there any other architectures to consider?

⏺ Web Search("vision transformer ViT MNIST PyTorch implementation small
            dataset performance 2024")
  ⎿  Did 1 search in 28s

⏺ Web Search("MobileNet EfficientNet lightweight neural networks MNIST
            performance 68k parameters PyTorch")
  ⎿  Did 3 searches in 36s

⏺ Based on my research, here are several other architectures worth
  considering for your ~68k parameter MNIST comparison:

  Modern Lightweight Architectures

  1. Vision Transformer (ViT)
  - Performance: ~98% on MNIST (but suboptimal for small images)
  - Challenge: Requires large patches or pre-training for small datasets
  - Available: Multiple PyTorch implementations from scratch

  2. MobileNet Variants
  - MobileNetV1/V2/V3: Depthwise separable convolutions for efficiency
  - Advantage: Designed for mobile/edge deployment
  - Parameter control: Adjustable width multipliers to hit 68k target

  3. EfficientNet-Tiny
  - Scaling approach: Compound scaling (depth, width, resolution)
  - Efficiency: Superior accuracy/parameter ratio
  - Customizable: Can be scaled down to 68k parameters

  Specialized Small-Scale Architectures

  4. Squeeze-and-Excitation Networks (SENet)
  - Addition: Channel attention mechanism
  - Benefit: Better feature representations with minimal overhead
  - Integration: Can enhance your existing CNN

  5. ShuffleNet
  - Innovation: Channel shuffling for group convolutions
  - Target: Ultra-lightweight applications
  - Performance: Competitive accuracy with fewer parameters

  6. Ghost Networks
  - Concept: Generate more features from cheap operations
  - Efficiency: Reduces computational cost significantly

  Hybrid Approaches

  7. CNN + Attention
  - Add lightweight attention mechanisms to your current CNN
  - Minimal parameter increase, potentially better performance

  Recommendation

  For fair comparison at 68k parameters, I'd suggest:
  1. MobileNetV2 (scaled down) - Modern depthwise separable convs
  2. EfficientNet-B0 (reduced) - State-of-the-art efficiency principles
  3. Your CNN + SE blocks - Enhance current architecture

  These would provide interesting architectural diversity while maintaining
  the parameter budget constraint!

### > What is the paper for EfficientNet-B0?

⏺ Web Search("EfficientNet B0 paper "EfficientNet: Rethinking Model Scaling" Mingxing Tan Quoc V. Le")
  ⎿  Did 1 search in 22s

⏺ The EfficientNet-B0 paper is:

  "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
  - Authors: Mingxing Tan, Quoc V. Le (Google Research)
  - Published: ICML 2019
  - arXiv: https://arxiv.org/abs/1905.11946

  Key Contributions

  1. Compound Scaling Method
  - Systematically scales depth, width, and resolution together
  - Uses simple compound coefficient rather than arbitrary scaling

  2. EfficientNet-B0 Baseline
  - Mobile-sized baseline designed with Neural Architecture Search (NAS)
  - Optimizes accuracy vs FLOPS tradeoff
  - Uses inverted bottleneck residual blocks (MBConv)

  3. EfficientNet Family
  - B0 through B7 variants using compound scaling on the B0 baseline
  - B7 achieves 84.4% ImageNet top-1 accuracy
  - 8.4x smaller and 6.1x faster than best existing ConvNets

  Architecture Details

  - 9 sequential stages of MBConv blocks
  - Compound scaling formula: depth × width^α × resolution^β
  - EfficientNet-B0: 5.3M parameters, optimized for mobile deployment

  This paper revolutionized efficient CNN design and established the
  principle that balanced scaling across all dimensions
  (depth/width/resolution) is more effective than scaling any single
  dimension.

### > MobileNetV2 also sounds interesting. Can it do convolutional invariance along one dimension but not the other?

⏺ Web Search("MobileNetV2 asymmetric convolution rectangular kernels different stride dimensions PyTorch")
  ⎿  Did 5 searches in 55s

⏺ Yes! MobileNetV2 can definitely handle asymmetric/rectangular
  convolutions, and PyTorch makes this straightforward. Here's what
  you need to know:

  PyTorch Conv2d Asymmetric Support

  Built-in rectangular kernels and different strides:
  # Different kernel dimensions and strides per axis
  nn.Conv2d(channels_in, channels_out,
            kernel_size=(height, width),    # e.g., (1, 7) or (5, 1)
            stride=(stride_h, stride_w),    # e.g., (1, 2)
            padding=(pad_h, pad_w))

  MobileNetV2 + Asymmetric Modifications

  Standard MobileNetV2 uses:
  - 3×3 depthwise convolutions
  - 1×1 pointwise convolutions

  For your use case, you could modify the depthwise layers:
  # Instead of (3,3) depthwise, use asymmetric:
  nn.Conv2d(channels, channels,
            kernel_size=(1, 7),      # Invariant along width, not height
            stride=(1, 1),           # Or (1, 2) for different stride behavior
            groups=channels)         # Depthwise separate groups - no "channel mixing"

  ACNet Approach

  There's also ACNet (Asymmetric Convolution Networks) which specifically addresses this:
  - Replaces square kernels with asymmetric convolution blocks
  - Uses 1D asymmetric convolutions to strengthen square kernels
  - Can be fused back for inference efficiency

  Your Real Project Application

  For "convolutional invariance along one dimension but not the other":

  Example - Time-series with spatial structure:
  - Height: Preserve spatial relationships (no invariance)
  - Width: Temporal invariance (pooling/large kernels)
  - Solution: Use (3,1) and (1,7) kernels strategically

  Implementation: Modify MobileNetV2's depthwise separable
  convolutions with rectangular kernels matching your invariance
  requirements.

### > why "groups=channels"?  Doesn't that inhibit "channel mixing" which adds to "modeling capacity"?

⏺ Excellent question! You're absolutely right to question this - it's a key architectural tradeoff in MobileNets.

  Why `groups=channels` (Depthwise Convolution)

  `groups=channels` means each input channel gets its own separate filter - no mixing between channels during spatial convolution.

  Example:
```python
  # Regular conv: 64 input channels → 64 output channels
  nn.Conv2d(64, 64, 3x3)  # 64×64×3×3 = 36,864 parameters

  # Depthwise conv: each channel processed independently
  nn.Conv2d(64, 64, 3x3, groups=64)  # 64×1×3×3 = 576 parameters
```
  The Tradeoff You Identified

  You're correct - depthwise convolution sacrifices channel mixing for efficiency:
  - Lost: Cross-channel spatial interactions
  - Gained: 64× fewer parameters, much faster

  MobileNet's Solution: Depthwise Separable Convolution

  MobileNet recovers the channel mixing with a two-step process:

  1. Depthwise (3×3, groups=channels): Spatial filtering per channel
  2. Pointwise (1×1, regular conv): Channel mixing across all channels

```python
  # MobileNet block
  nn.Conv2d(in_ch, in_ch, 3, groups=in_ch)     # Spatial, no mixing
  nn.Conv2d(in_ch, out_ch, 1)                  # Channel mixing, no spatial
```

  Why This Works

  Hypothesis: You can separate these operations:
  - Spatial relationships (depthwise 3×3)
  - Channel relationships (pointwise 1×1)

  Result: Similar modeling capacity with ~8-9× fewer parameters than regular convolution.

  The Capacity Question

  You're right that there's some capacity loss - but MobileNet compensates with:
  - Inverted residuals (expand → depthwise → compress)
  - More layers (deeper networks)
  - Clever scaling (width/depth multipliers)

  The efficiency gains usually outweigh the capacity loss for mobile applications!
