import torch
import torch.nn as nn

from src.models.base_model import BaseDLModel

ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "LeakyReLU": nn.LeakyReLU,
}


class CustomizableSleepCAM(nn.Module, BaseDLModel):
    def __init__(self, input_channels, activation, num_classes=5):
        super(CustomizableSleepCAM, self).__init__()
        activation_class = ACTIVATIONS.get(activation)

        self.conv1 = nn.Conv1d(input_channels, 96, kernel_size=50, padding=25)
        self.elu1 = activation_class()
        self.conv2 = nn.Conv1d(96, 96, kernel_size=10, padding=5)
        self.elu2 = activation_class()
        self.maxpool = nn.MaxPool1d(kernel_size=10)
        self.layernorm = nn.LayerNorm(96)

        self.conv3 = nn.Conv1d(96, 96, kernel_size=5, padding=2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(96, num_classes)

    def forward(self, x):
        """
        x: (B, N, T, C)
            B: Batch size
            N: Number of epochs
            T: Number of samples (length of each epoch)
            C: Number of channels
        Output: (B, num_classes)
        """
        assert x.dim() == 4, f"Expected (B, N, T, C), got {x.shape}"
        B, N, T, C = x.shape

        x = x.reshape(B, N * T, C)
        x = x.transpose(1, 2)  # (B, C, N*T)

        # Extract features
        x = self.elu1(self.conv1(x))
        x = self.elu2(self.conv2(x))
        x = self.maxpool(x)

        x = x.transpose(1, 2)  # (B, seq_len, features=96)
        x = self.layernorm(x)
        x = x.transpose(1, 2)  # (B, 96, seq_len)

        x = self.conv3(x)

        x = self.global_avg_pool(x).squeeze(-1)  # (B, 96)

        x = self.classifier(x)  # (B, num_classes)

        return x

    @classmethod
    def load_model(cls, model_path, config):
        model = cls(**config)
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
