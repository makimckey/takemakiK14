import torch.nn as nn

LOSS_DICT = {
    "cross_entropy": nn.CrossEntropyLoss,
}


class LossSelector:
    @staticmethod
    def select(name: str):
        loss_class = LOSS_DICT.get(name)
        if loss_class is None:
            raise ValueError(f"Loss '{name}' is not defined in `LOSS_DICT`.")
        return loss_class
