import math
from pathlib import Path

import torch
from loguru import logger

from .base_callback import BaseCallback


class ModelCheckpoint(BaseCallback):
    def __init__(self, output_dir, filename, monitor, mode):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
        self.filename = filename

        self.last_saved_weight_path = None

        self.monitor = monitor
        if mode == "min":
            self.best_val = math.inf
            self.is_improved = lambda v: v < self.best_val
        else:
            self.best_val = -math.inf
            self.is_improved = lambda v: v > self.best_val

    def on_train_loop_end(self, logs):
        current_val = logs["metrics"][self.monitor]
        if not self.is_improved(current_val):
            return

        # Save the current model weight
        checkpoint = {
            "epoch": logs["epoch"],
            "model_state_dict": logs["model"].state_dict(),
            "optimizer_state_dict": logs["optimizer"].state_dict(),
        }
        save_filename = self._render_filename(logs)
        save_path = self.output_dir / save_filename
        torch.save(checkpoint, save_path)

        logger.info(
            f"Epoch {logs['epoch']}, {self.monitor} reached {current_val:.4f} (best: {self.best_val:.4f}), "
            f"saving model to `{save_path}`"
        )
        self.remove_old_checkpoint(new_model_path=save_path)
        self.best_val = current_val
        self.last_saved_weight_path = save_path

    def _format_context(self, logs: dict) -> dict:
        """Build a dict for str.format from logs and logs['metrics']."""
        ctx = {}
        # shallow copy of logs except large/unsuitable objects
        for k, v in logs.items():
            if k not in {"model", "optimizer"}:
                ctx[k] = v
        # metrics flatten
        metrics = logs.get("metrics") or {}
        ctx.update(metrics)
        return ctx

    def _render_filename(self, logs: dict) -> str:
        """Render the filename from the template and context."""
        ctx = self._format_context(logs)
        try:
            name = self.filename.format(**ctx)
        except KeyError as e:
            available = ", ".join(sorted(ctx.keys()))
            raise KeyError(
                f"Filename placeholder {{{e.args[0]}}} not found in logs/metrics. "
                f"Available keys: {available}"
            )
        return name

    def remove_old_checkpoint(self, new_model_path):
        old_model_path = self.last_saved_weight_path

        # Skip the process if there's no old weight
        if not old_model_path:
            return
        if not old_model_path.exists():
            return

        # Skip if both paths are identical
        # — otherwise the new model would overwrite the old one
        if old_model_path.resolve() == new_model_path.resolve():
            return

        try:
            old_model_path.unlink()
            logger.info(f"Removed old checkpoint: {old_model_path}")
        except Exception as e:
            logger.exception(f"Failed to remove old checkpoint: {e}")
