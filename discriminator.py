import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, image_channel =3):
        super(Discriminator, self).__init__() 
        
        self.model = nn.Sequential(
            nn.Conv2d(image_channel, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256,1,3,1,1),
            
            nn.Flatten(),
            nn.Linear(8*8,1), #4*4
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.model(x)
        return x