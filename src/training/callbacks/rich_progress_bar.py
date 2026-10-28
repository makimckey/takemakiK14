from .base_callback import BaseCallback

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)


class RichProgressBar(BaseCallback):
    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, logs):
        self.columns = [
            TextColumn("[bold cyan]{task.fields[desc]}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn(" "),
            TextColumn("loss: {task.fields[loss]:.4f}"),
            TextColumn(" acc: {task.fields[acc]:.4f}"),
            TextColumn(" "),
            TimeElapsedColumn(),
            TextColumn(" / "),
            TimeRemainingColumn(),
        ]
        self.task_desc = logs["description"]
        self.progress = Progress(*self.columns, transient=True)
        self.progress.start()

        self.task_id = self.progress.add_task(
            self.task_desc,
            total=logs["total_batchs"],
            loss=0,
            acc=0,
            desc=self.task_desc,
        )

    def on_batch_end(self, logs):
        self.progress.update(
            self.task_id,
            advance=1,
            loss=logs["loss"],
            acc=logs["acc"],
            desc=self.task_desc,
        )

    def on_epoch_end(self, logs):
        self.progress.stop()
