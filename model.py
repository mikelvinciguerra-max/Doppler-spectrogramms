import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=5):
        super(CNN, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=5),
            nn.Mish(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.Mish(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.Mish(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),

            nn.Dropout(p=0.2),
            nn.Flatten(),

            nn.LazyLinear(out_features=256),
            nn.Mish(),

            nn.Dropout(p=0.3),

            nn.Linear(in_features=256, out_features=128),
            nn.Mish(),

            nn.Linear(in_features=128, out_features=num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.nn(x)