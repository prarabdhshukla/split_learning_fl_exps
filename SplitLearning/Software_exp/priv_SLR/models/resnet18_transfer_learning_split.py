import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

model = models.resnet18(pretrained=True)
model_list = list(model.children())

front_layers = model_list[:4]
center_layers = model_list[4:9]
back_layers = model_list[9:]

class front(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = front_layers[0]
        self.bn1 = front_layers[1]
        self.relu = front_layers[2]
        self.maxpool = front_layers[3]


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x

class center(nn.Module):

    def __init__(self):

        super().__init__()

        self.layer1 = center_layers[0]
        self.layer2 = center_layers[1]
        self.layer3 = center_layers[2]
        self.layer4 = center_layers[3]
        self.avgpool = center_layers[4]

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        return x

class back(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(512, 10)    #not using pretrained last layer weights because pretrained model has final layer dimension as 512 * 1000 (1k imagenet)

    def forward(self,x):
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x





