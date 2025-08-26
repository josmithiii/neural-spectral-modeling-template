"""Custom callback for grouped metric display."""

from typing import Any, Dict, Optional
from lightning.pytorch.callbacks import Callback
from rich.console import Console
from rich.table import Table
import torch


class GroupedMetricsDisplayCallback(Callback):
    """Callback that displays test metrics grouped by parameter."""

    def __init__(self):
        super().__init__()
        self._console = Console()

    def on_test_end(self, trainer, pl_module) -> None:
        """Display grouped test metrics table after testing ends."""
        self._display_grouped_metrics(trainer)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Display grouped test metrics table after test epoch ends."""
        self._display_grouped_metrics(trainer)


    def _display_grouped_metrics(self, trainer) -> None:
        """Display grouped test metrics table."""
        # Get the test metrics
        metrics = trainer.logged_metrics
        test_metrics = {k: v for k, v in metrics.items() if k.startswith('test/')}

        if not test_metrics:
            return


        # Group metrics by parameter
        parameter_groups = {}
        for metric_name, value in test_metrics.items():
            # Remove 'test/' prefix
            clean_name = metric_name.replace('test/', '')

            # Extract parameter name (before _acc)
            if '_acc' in clean_name:
                param_name = clean_name.split('_acc')[0]
            elif clean_name == 'loss':
                param_name = 'loss'
            else:
                param_name = clean_name

            if param_name not in parameter_groups:
                parameter_groups[param_name] = []
            parameter_groups[param_name].append((clean_name, value))

        # Create the main table with Lightning's styling
        table = Table(
            title="Test Results (Grouped by Parameter)",
            title_style="bold magenta",
            show_header=True,
            header_style="bold magenta",
            border_style="bright_blue"
        )
        table.add_column("Test metric", style="cyan", no_wrap=True)
        table.add_column("DataLoader 0", justify="center", style="green")

        # Get all parameter names sorted
        all_params = sorted(parameter_groups.keys())


        for i, param_name in enumerate(all_params):
            # Add metrics for this parameter
            for metric_name, value in parameter_groups[param_name]:
                if isinstance(value, torch.Tensor):
                    value = value.item()
                formatted_value = f"{value}" if isinstance(value, (int, float)) else str(value)
                table.add_row(f"test/{metric_name}", formatted_value)

            # Add separator between groups (except after the last group)
            if i < len(all_params) - 1:
                table.add_row("", "")

        # Print the grouped table
        self._console.print(table)
