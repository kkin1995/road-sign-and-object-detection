import torch.nn as nn

# (3, 480, 816)


class TrafficSignNet(nn.Module):
    def __init__(self):
        super(TrafficSignNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3)),  # (3, 478, 814)
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),  # (64, 476, 812)
            nn.MaxPool2d((2, 2)),  # (64, 238, 406)
            nn.ReLU(),
            # nn.Conv2d(64, 64, (3, 3)),  # (3, 394, 674)
            nn.Flatten(),
            # nn.Linear(64 * (400 - 6) * (680 - 6), 4),
            nn.Linear(64 * 238 * 406, 4),
        )

    def forward(self, x):
        return self.model(x)
