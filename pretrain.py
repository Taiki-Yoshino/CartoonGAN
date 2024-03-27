import torch
import torch.nn as nn
from generator import Generator
from torch import optim
from torchvision import utils as vutils
import matplotlib.pyplot as plt
import numpy as np

def psnr(data1, data2):
    mse = torch.mean((data1 - data2)**2, dim=[1,2,3])
    max_pixel = 1.0
    psnr = 20* torch.log10(max_pixel/torch.sqrt(mse))
    psnr = torch.mean(psnr).numpy()
    return np.round(psnr*10000)/ 10000

def fit(real_images_loader,  epochs, lr, beta1, device = "cuda:0"):
    generator = Generator().to(device)
    content_loss = nn.L1Loss().to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

    for epoch in range(epochs):
        for i, real_images in enumerate(real_images_loader):
            real_images = real_images.to(device)
            
            #G
            generator.zero_grad()
            generated_images = generator(real_images)
            loss_g = content_loss(generated_images, real_images)
            loss_g.backward()
            optimizer_G.step()
           
            if((epoch % 1 == 0) and (i == 0)): 
                print(f"Epoch [{epoch}/{epochs}] Avg_PSNR: {psnr(generated_images.detach().cpu(), real_images.detach().cpu())}")
                i += 1

        if epoch == (epochs-1):  
            fake_img_grid = vutils.make_grid(generated_images.detach().cpu(), padding=2, normalize=True)
            real_img_grid = vutils.make_grid(real_images.detach().cpu(), padding=2, normalize=True)

            plt.figure(figsize=(16,30))
            plt.imshow(real_img_grid.permute(1, 2, 0).squeeze())
            plt.axis(False)
            plt.title("original real-world images")
            plt.show()
            
            plt.figure(figsize=(16,30))
            plt.imshow(fake_img_grid.permute(1, 2, 0).squeeze())
            plt.axis(False)
            plt.title("generated real-world images")
            plt.show()
            
    
    torch.save(generator, 'pretrained_generator.pth')
    return generator
