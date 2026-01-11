import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.n_classes = n_classes
        # input: 3x32x32    
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=(0,1)) # 32x30x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=(0,1)) # 32x28x32
        self.conv2_2 = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=(1,1)) # 32x28x32
        self.bn1 = nn.BatchNorm2d(96)
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x14x16
        
        self.conv3 = nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=(0,1)) # 128x12x16 
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=(0,1)) # 128x10x16
        self.conv4_2 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=(1,1)) # 128x10x16
        self.bn2 = nn.BatchNorm2d(384)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2) # 128x5x8
        self.shorcutConv2 = nn.Conv2d(96, 384, kernel_size=5, stride=1, padding=(0,2))
        
        self.conv5 = nn.Conv2d(384, 512, kernel_size=3, stride=1, padding=(0,1)) # 502x3x8
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=(0,1)) # 502x1x8
        self.bn3 = nn.BatchNorm2d(1024)

        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(0.1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024 * 8, 1024)
        self.fc2 = nn.Linear(1024, n_classes)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.bn1(self.conv2_2(x)))
        x = self.down1(x)

        shortcut = x.clone()
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.bn2(self.conv4_2(x)))
        x = self.down2(x)
        shortcut = self.relu(self.bn2(self.shorcutConv2(shortcut)))
        x += self.down2(shortcut)

        x = self.relu(self.conv5(x))
        x = self.relu(self.bn3(self.conv6(x)))

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        out = self.fc2(x)

        return out

def init_params(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
