import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions


def flatten(x):
    N = x.shape[0]  # read in N, C, H, W

    x = x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

    return x


class ThreeLayerConvNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        in_channel = 3
        num_classes = 10
        channel_1 = 32
        channel_2 = 16

        self.conv1 = nn.Conv2d(in_channel, channel_1, (kwargs['filter_size_1'], kwargs['filter_size_1']),
                               padding=int(kwargs['filter_size_1']/2))
        nn.init.kaiming_normal_(self.conv1.weight)

        self.conv2 = nn.Conv2d(channel_1, channel_2, (3, 3), padding=1)
        nn.init.kaiming_normal_(self.conv1.weight)

        self.fc = nn.Linear(channel_2 * 32 * 32, num_classes, bias=True)
        nn.init.kaiming_normal_(self.fc.weight)

    @classmethod
    def get_name(cls):
        return 'three_layer_conv_net'

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = flatten(x)
        scores = self.fc(x)

        return scores