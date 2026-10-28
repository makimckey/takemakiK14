import time

from loguru import logger

from .base_callback import BaseCallback


class SimpleLogger(BaseCallback):
    def __init__(self):
        super().__init__()

    def on_train_loop_begin(self, logs):
        self.start_time = time.time()

    def on_train_loop_end(self, logs):
        elapsed = time.time() - self.start_time
        metrics = {k: f"{v:.4f}" for k, v in logs["metrics"].items()}
        info = {
            "Epoch": f"{logs['epoch']}/{logs['total_epochs']}",
            "Time": f"{elapsed:.2f}s",
            **metrics,
        }
        info_str = [f"{k}: {v}" for k, v in info.items()]
        logger.info(", ".join(info_str))

    def on_evaluate_begin(self, logs):
        self.start_time = time.time()

    def on_evaluate_end(self, logs):
        elapsed = time.time() - self.start_time
        metrics = {k: f"{v:.4f}" for k, v in logs["metrics"].items()}
        info = {
            "Time": f"{elapsed:.2f}s",
            **metrics,
        }
        info_str = [f"{k}: {v}" for k, v in info.items()]
        logger.info(", ".join(info_str))
