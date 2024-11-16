# -*- coding: utf-8 -*-
"""
Author: Lok Yee Joey Cheung
Code for implementating GAN for sythetic brain images generation.

Reference: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html 
"""

import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt


#A new class that inherits from PyTorch's Dataset class
class BrainImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir # image's directory 
        self.transform = transform 
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image

#Kernel: capture spatial details (smaller, finer details); Stride: upsampling factor (larger, higher dim) ; padding: subtract pixels around border, determine output dim
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0, bias=False), # latent vector dim = 100
            nn.BatchNorm2d(1024),   #Re-centering, stablize training, accelerate convergence of gradient 
            nn.ReLU(True), #introduces non-linearity
            # output_size = (input_size - 1) * stride + kernel_size - 2 * padding
            # State size. (1024, 4, 4)

            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State size. (512, 8, 8)

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State size. (256, 16, 16)

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State size. (128, 32, 32)

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State size. (64, 64, 64)

            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh() #Since images are normalized to the range (-1, 1)
            # Output size. (1, 128, 128)
        )

    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``1 x 64 x 64``
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            # LeakyReLU: Address the vanishing gradient problem, improve training stability; negative slope parameter =0.2 allows the network to learn even when the input is negative.
            nn.LeakyReLU(0.2, inplace=True), 
            # Output size: (Input size−Kernel size+2×Padding)/Stride +1
            # state size. ``64 x 32 x 32``

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``128 x 16 x 16``

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``256 x 8 x 8``

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``512 x 4 x 4``

            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``1024x 2 x 2``

            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output size. (1, 1, 1)
        )

    def forward(self, input):
        return self.main(input).view(-1)


# Define image size and transformations
image_size = 128
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] centers the data around zero to stablize training
])

train_dir = '/home/groups/comp3710/OASIS/keras_png_slices_train'.strip()
validate_dir = '/home/groups/comp3710/OASIS/keras_png_slices_validate'.strip()
test_dir = ' /home/groups/comp3710/OASIS/keras_png_slices_test'.strip()

# Load the datasets
# Dataloader is used for batch processing & parallel data loading with workers
train_data = BrainImageDataset(image_dir=train_dir , transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True) #batch_size=64: less GPU memory, good generalization, wont converge too quickly 

val_data = BrainImageDataset(image_dir=validate_dir, transform=transform)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

test_data = BrainImageDataset(image_dir=test_dir, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the generator and discriminator
netG = Generator().to(device)
netD = Discriminator().to(device)

# Define Loss function and Optimizers https://arxiv.org/pdf/1511.06434 
criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999)) #0.5 momentum (beta) helps stabilize training
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999)) #0.999 (beta2) moving average of the squared gradients 

# Define learning rate schedulers to stabilize the training process
schedulerD = lr_scheduler.StepLR(optimizerD, step_size=10, gamma=0.5)  # LR*0.5 every 10 epochs
schedulerG = lr_scheduler.StepLR(optimizerG, step_size=10, gamma=0.5)

# Training loop
num_epochs = 50
#fixed_noise = torch.randn(64, 100, 1, 1, device=device)

# Define the random noise latent vector size, size of input to produce synthetic data
nz = 100

# Lists to store loss values
G_losses = []
D_losses = []

for epoch in range(num_epochs):
    netG.train()
    netD.train()
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

    for i, data in enumerate(progress_bar):
        # Train Discriminator with real images
        netD.zero_grad()
        real_images = data.to(device)
        batch_size = real_images.size(0)
        label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
        output = netD(real_images).view(-1) # Forward pass real batch through D
        errD_real = criterion(output, label) # Calculate loss on all-real batch
        # Calculate gradients for D in backward pass
        errD_real.backward() 
        D_x = output.mean().item() 

        # Train with fake images
        noise = torch.randn(batch_size, nz, 1, 1, device=device) # Generate batch of random noise latent vectors used as input to netG 
        #Generate fake image batch with G
        fake = netG(noise)
        label.fill_(0)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1) 
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label) 
        # Calculate the gradients for this batch, summed with previous gradients in this iteration
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()


        # Train Generator
        netG.zero_grad()
        # Set fake images to real to trick D
        label.fill_(1)  
        #Perform forward pass to compute the discriminator's output for the fake images generated by G.
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step() # Update G

        # Store the losses
        D_losses.append(errD.item())
        G_losses.append(errG.item())

        progress_bar.set_postfix(Loss_D=f"{errD.item():.4f}", Loss_G=f"{errG.item():.4f}")
    
    # Step the schedulers after each epoch
    schedulerD.step()
    schedulerG.step()

    # Plot and save losses after each epoch
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plot.png") 
    plt.close() 

with torch.no_grad():
    noise = torch.randn(64, 100, 1,1, device=device)
    fake_images = netG(noise)

    # Visualize and save some generated images
    for j in range(2):
        plt.figure(figsize=(8, 8))
        plt.imshow(fake_images[j].cpu().squeeze(), cmap='gray')
        plt.axis('off')
        plt.savefig(f"generated_image_{j}.png")
        plt.close()
