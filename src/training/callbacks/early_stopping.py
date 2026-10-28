from .base_callback import BaseCallback

from loguru import logger


class EarlyStopping(BaseCallback):
    def __init__(self, patience: int, monitor: str, mode: str):
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.bad_epochs = 0

        if mode == "min":
            self.best = float("inf")
            self.is_improved = lambda v: v < self.best
        else:
            self.best = -float("inf")
            self.is_improved = lambda v: v > self.best

    def on_train_loop_end(self, logs):
        # If the score is improved, just reset the count
        current_val = logs["metrics"][self.monitor]
        if self.is_improved(current_val):
            self.bad_epochs = 0
            self.best = current_val
            return

        # Count the bad epochs
        self.bad_epochs += 1
        if self.bad_epochs < self.patience:
            return

        # If the number of bad_epochs reaches the patience, break the training loop
        logger.info(
            f"Stopping threshold reached:"
            f"{self.monitor} did not improve in the last {self.bad_epochs} epochs, signaling Trainer to stop."
        )
        return {"should_stop": True}
