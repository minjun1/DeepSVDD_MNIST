import torch
import torch.nn as nn
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.dense1 = nn.Linear(7*7*32, 128)

    def forward(self, x):
        x = F.elu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.elu(self.bn3(self.conv3(x)))
        x = x.view(-1, 7*7*32)
        x = F.elu(self.dense1(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense2 = nn.Linear(128, 7*7*32)
        self.deconv1 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(8)

    def forward(self, x):
        x = F.elu(self.dense2(x))
        x = x.view(-1, 32, 7, 7)
        x = F.elu(self.bn4(self.deconv1(x)))
        x = F.elu(self.bn5(self.deconv2(x)))
        x = torch.sigmoid(self.deconv3(x))
        return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
