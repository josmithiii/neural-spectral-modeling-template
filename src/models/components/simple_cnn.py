from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """Lightweight convolutional network for VIMH spectrogram inputs.

    Supports:
      - classification (single or multi-head)
      - regression (single or multi-head, with sigmoid outputs in [0, 1])
      - optional auxiliary scalar inputs
    """

    def __init__(
        self,
        input_channels: int = 1,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        fc_hidden: int = 128,
        output_size: Optional[int] = None,
        heads_config: Optional[Dict[str, int]] = None,
        dropout: float = 0.25,
        input_size: int = 28,
        output_mode: str = "classification",  # "classification" or "regression"
        parameter_names: Optional[List[str]] = None,
        parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        auxiliary_input_size: int = 0,
        auxiliary_hidden_size: int = 32,
    ) -> None:
        super().__init__()

        # ---- store basic config ----
        self.output_mode = output_mode
        self.parameter_names = parameter_names or []
        self.parameter_ranges = parameter_ranges or {}
        self.auxiliary_input_size = auxiliary_input_size
        self.auxiliary_hidden_size = auxiliary_hidden_size
        self.fc_hidden = fc_hidden
        self.input_size = input_size
        self.input_resolution = (input_size, input_size)

        # Normalize heads_config to a dict (may be empty, meaning "auto-configure later")
        if heads_config is None:
            heads_config = {}
        self.heads_config: Dict[str, int] = heads_config
        self.is_multihead: bool = len(heads_config) > 1

        # ---- convolutional front-end ----
        # After two MaxPool2d(stride=2), size becomes input_size / 4
        pooled_size = input_size // 4
        if pooled_size == 7:          # 28 input -> 7 after pooling
            self.adaptive_pool_size = (7, 7)
        elif pooled_size == 8:        # 32 input -> 8 after pooling
            self.adaptive_pool_size = (4, 4)  # 8 divisible by 4
        else:
            self.adaptive_pool_size = (4, 4)  # safe default

        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, conv1_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv1_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second conv block
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv2_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Adaptive pooling
            nn.AdaptiveAvgPool2d(self.adaptive_pool_size),
        )

        linear_input_size = (
            conv2_channels * self.adaptive_pool_size[0] * self.adaptive_pool_size[1]
        )

        self.shared_features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- auxiliary input branch (optional) ----
        self.auxiliary_net: Optional[nn.Module]
        if auxiliary_input_size > 0:
            self.auxiliary_net = nn.Sequential(
                nn.Linear(auxiliary_input_size, auxiliary_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(auxiliary_hidden_size, auxiliary_hidden_size),
                nn.ReLU(),
            )
            combined_feature_size = fc_hidden + auxiliary_hidden_size
        else:
            self.auxiliary_net = None
            combined_feature_size = fc_hidden

        self._combined_feature_size = combined_feature_size

        # ---- output heads / classifier ----
        # Always define a heads ModuleDict so auto-config code can fill it later.
        self.heads = nn.ModuleDict()

        if len(self.heads_config) > 0:
            # We have a heads_config from the config/metadata: build heads now.
            self._build_heads(self.heads_config)
        else:
            # No heads_config yet -> single-head fallback.
            # This is mainly for backward compatibility and trivial experiments.
            if self.output_mode == "regression":
                # Single regression vector; by default length = len(parameter_names) or 1.
                out_dim = len(self.parameter_names) if self.parameter_names else 1
                self.classifier = nn.Sequential(
                    nn.Linear(combined_feature_size, out_dim),
                    nn.Sigmoid(),
                )
            else:
                # Single classification head; default to output_size or 1
                num_classes = output_size if output_size is not None else 1
                self.classifier = nn.Linear(combined_feature_size, num_classes)

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, auxiliary: Optional[torch.Tensor] = None):
        """Forward pass.

        Args:
            x: (batch, C, H, W) spectrogram tensor.
            auxiliary: Optional (batch, auxiliary_input_size) tensor.

        Returns:
            - If heads are defined: dict[str, Tensor] with shape (batch, head_dim)
            - Else: Tensor with shape (batch, out_dim) from self.classifier
        """
        x = self.conv_layers(x)
        shared_features = self.shared_features(x)

        if self.auxiliary_net is not None and auxiliary is not None:
            aux_features = self.auxiliary_net(auxiliary)
            features = torch.cat([shared_features, aux_features], dim=1)
        else:
            features = shared_features

        # Prefer multi-head (or single-head via heads_config) if heads are present
        if hasattr(self, "heads") and len(self.heads) > 0:
            return {name: head(features) for name, head in self.heads.items()}

        # Fall back to single classifier
        if not hasattr(self, "classifier"):
            raise RuntimeError(
                "SimpleCNN is misconfigured: no heads and no classifier. "
                "Did you forget to call _build_heads or provide heads_config?"
            )
        return self.classifier(features)

    # -------------------------------------------------------------------------
    # Helpers for auto-configuration from dataset metadata
    # -------------------------------------------------------------------------
    def _build_heads(self, heads_config: Dict[str, int]) -> None:
        """(Re)build heads for auto-configuration.

        For regression:
            each head -> Sequential(Linear(combined_features, 1), Sigmoid())
        For classification:
            each head -> Linear(combined_features, num_classes)
        """
        self.heads_config = heads_config
        self.is_multihead = len(heads_config) > 1

        combined_feature_size = self._combined_feature_size

        if self.output_mode == "regression":
            self.heads = nn.ModuleDict(
                {
                    head_name: nn.Sequential(
                        nn.Linear(combined_feature_size, 1),
                        nn.Sigmoid(),
                    )
                    for head_name in heads_config.keys()
                }
            )
        else:
            self.heads = nn.ModuleDict(
                {
                    head_name: nn.Linear(combined_feature_size, num_classes)
                    for head_name, num_classes in heads_config.items()
                }
            )

    def _rebuild_auxiliary_and_heads(self) -> None:
        """Rebuild auxiliary_net and heads if auxiliary_input_size changes."""
        # Rebuild auxiliary network
        if self.auxiliary_input_size > 0:
            if self.auxiliary_net is None:
                self.auxiliary_net = nn.Sequential(
                    nn.Linear(self.auxiliary_input_size, self.auxiliary_hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.25 / 2),
                    nn.Linear(self.auxiliary_hidden_size, self.auxiliary_hidden_size),
                    nn.ReLU(),
                )
            else:
                first_layer = self.auxiliary_net[0]
                if (
                    hasattr(first_layer, "in_features")
                    and first_layer.in_features != self.auxiliary_input_size
                ):
                    self.auxiliary_net[0] = nn.Linear(
                        self.auxiliary_input_size, self.auxiliary_hidden_size
                    )
            combined_feature_size = self.fc_hidden + self.auxiliary_hidden_size
        else:
            self.auxiliary_net = None
            combined_feature_size = self.fc_hidden

        self._combined_feature_size = combined_feature_size

        # Rebuild heads with the new combined feature size
        if hasattr(self, "heads_config") and len(self.heads_config) > 0:
            if self.output_mode == "regression":
                self.heads = nn.ModuleDict(
                    {
                        head_name: nn.Sequential(
                            nn.Linear(combined_feature_size, 1),
                            nn.Sigmoid(),
                        )
                        for head_name in self.heads_config.keys()
                    }
                )
            else:
                self.heads = nn.ModuleDict(
                    {
                        head_name: nn.Linear(combined_feature_size, num_classes)
                        for head_name, num_classes in self.heads_config.items()
                    }
                )
        else:
            # No heads_config -> rebuild single-head classifier
            if self.output_mode == "regression":
                out_dim = len(self.parameter_names) if self.parameter_names else 1
                self.classifier = nn.Sequential(
                    nn.Linear(combined_feature_size, out_dim),
                    nn.Sigmoid(),
                )
            else:
                # default to 1 logit if nothing else is known
                self.classifier = nn.Linear(combined_feature_size, 1)


if __name__ == "__main__":
    # Quick smoke tests using VIMH-style spectrogram tensors
    batch = torch.randn(2, 1, 32, 32)  # Batch of 32x32 spectrograms

    # Multihead classification example
    model_multi_cls = SimpleCNN(
        input_channels=1,
        heads_config={"digit": 10, "other_head": 5},
        input_size=32,
        output_mode="classification",
    )
    out_multi_cls = model_multi_cls(batch)
    print("Multihead classification:")
    for name, tensor in out_multi_cls.items():
        print(f"  {name}: {tensor.shape}")

    # Multihead regression example
    model_multi_reg = SimpleCNN(
        input_channels=1,
        heads_config={"midi_pitch": 1},
        input_size=32,
        output_mode="regression",
    )
    out_multi_reg = model_multi_reg(batch)
    print("\nMultihead regression:")
    for name, tensor in out_multi_reg.items():
        print(f"  {name}: {tensor.shape}")

    # Single-head regression fallback
    model_single_reg = SimpleCNN(
        input_channels=1,
        heads_config={},        # no heads yet
        input_size=32,
        output_mode="regression",
        parameter_names=["midi_pitch"],
    )
    out_single_reg = model_single_reg(batch)
    print("\nSingle-head regression fallback:", out_single_reg.shape)
