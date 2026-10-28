class BaseCallback:
    def __init__(self):
        pass

    def on_train_begin(self, logs):
        pass

    def on_train_end(self, logs):
        pass

    def on_evaluate_begin(self, logs):
        pass

    def on_evaluate_end(self, logs):
        pass

    def on_train_loop_begin(self, logs):
        pass

    def on_train_loop_end(self, logs):
        pass

    def on_epoch_begin(self, logs):
        pass

    def on_epoch_end(self, logs):
        pass

    def on_batch_begin(self, logs):
        pass

    def on_batch_end(self, logs):
        pass
