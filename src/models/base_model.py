from abc import ABC, abstractmethod


class BaseDLModel(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        NotImplementedError()

    @classmethod
    @abstractmethod
    def load_model(cls, model_path, config):
        NotImplementedError()
