import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions


class ThreeLayerConvNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        in_channel = 3
        num_classes = 10
        channel_1 = kwargs['channel_1']
        channel_2 = kwargs['channel_2']
        image_height = 32
        image_width = 32

        self.conv1 = nn.Conv2d(in_channel, channel_1, (kwargs['filter_size_1'], kwargs['filter_size_1']),
                               padding=int(kwargs['filter_size_1'] / 2))
        self.conv2 = nn.Conv2d(channel_1, channel_2, (3, 3), padding=1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.fc = nn.Linear(channel_2 * image_height * image_width, num_classes, bias=True)

        self.flatten = nn.Flatten()

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.fc.weight)

    @classmethod
    def get_name(cls):
        return 'three_layer_conv_net'

    def forward(self, x):
        if not x.is_cuda:
            x = x.to(device='cuda')

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.flatten(x)
        scores = self.fc(x)

        return scores


class ThreeLayerConvNetPool(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        in_channel = 3
        num_classes = 10
        channel_1 = 32
        channel_2 = 16
        image_height = 32
        image_width = 32

        self.conv1 = nn.Conv2d(in_channel, channel_1, (kwargs['filter_size_1'], kwargs['filter_size_1']),
                               padding=int(kwargs['filter_size_1'] / 2))
        self.conv2 = nn.Conv2d(channel_1, channel_2, (3, 3), padding=1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()

        in_features = int(channel_2 * image_height / 4 * image_width / 4)
        self.fc = nn.Linear(in_features, num_classes, bias=True)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.fc.weight)

    @classmethod
    def get_name(cls):
        return 'three_layer_conv_net_pool'

    def forward(self, x):
        if not x.is_cuda:
            x = x.to(device='cuda')

        x = self.pool1(self.conv1(x))
        x = self.relu1(x)

        x = self.pool2(self.conv2(x))
        x = self.relu2(x)

        x = self.flatten(x)
        scores = self.fc(x)

        return scores


class ThreeLayerConvNetBatch(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        in_channel = 3
        num_classes = 10
        channel_1 = 32
        channel_2 = 16
        image_height = 32
        image_width = 32

        self.conv1 = nn.Conv2d(in_channel, channel_1, (kwargs['filter_size_1'], kwargs['filter_size_1']),
                               padding=int(kwargs['filter_size_1'] / 2))

        self.conv2 = nn.Conv2d(channel_1, channel_2, (3, 3), padding=1)
        self.conv2_bn = nn.BatchNorm2d(channel_1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.fc_bn = nn.BatchNorm1d(channel_2 * image_height * image_width)
        self.fc = nn.Linear(channel_2 * image_height * image_width, num_classes, bias=True)

        self.flatten = nn.Flatten()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @classmethod
    def get_name(cls):
        return 'three_layer_conv_net_batch'

    def forward(self, x):
        if not x.is_cuda:
            x = x.to(device='cuda')

        x = self.conv1(x)

        x = self.relu1(x)
        x = self.conv2_bn(x)
        x = self.conv2(x)

        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc_bn(x)
        scores = self.fc(x)

        return scores


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


