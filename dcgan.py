# Adapted and extended from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

from __future__ import print_function
#%matplotlib inline
import click
from numpy.random.mtrand import beta
import wandb
import argparse
import os
import random
from PIL import Image 
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import pytorch_fid.fid_score
from numpy import random


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu, nz, nc, features_gen):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, features_gen * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_gen * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(features_gen * 8, features_gen * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_gen * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( features_gen * 4, features_gen * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_gen * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( features_gen * 2, features_gen, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_gen),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( features_gen, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, features_disc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, features_disc, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(features_disc, features_disc * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_disc * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(features_disc * 2, features_disc * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_disc * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(features_disc * 4, features_disc * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_disc * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(features_disc * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

@click.command()
@click.option('--num_epochs', default=5, help='Number of epochs to train.')
@click.option('--report_wandb', default=False, help='Use weights & biases for reporting.')
@click.option('--calculate_fid', default=False, help='Calculate the Frechet inception distance metric between fakes and reals.')
def train(num_epochs, report_wandb, calculate_fid):
    
    runid = random.randint(9999999)
    # Set random seed for reproducibility
    manualSeed = 222
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Number of FID samples
    fid_samples = 2048
    # Number of workers for dataloader
    workers = 1
    # Batch size during training
    batch_size = 128
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64
    # Number of channels in the training images. For color images this is 3
    nc = 3
    # Size of z latent vector (i.e. size of generator input)
    nz = 100
    # Size of feature maps in generator
    features_gen = 64
    # Size of feature maps in discriminator
    features_disc = 64
    # Learning rate for optimizers
    lr = 0.0002
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    tforms = transforms.Compose([
                                transforms.Grayscale(num_output_channels=3),
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5)),
                            ])
    dataset = dset.MNIST(root="dataset/", transform=tforms, download=True)

    # Create the dataloaders
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
    fid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create the generator
    netG = Generator(ngpu, nz, nc, features_gen).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)
    
    # Create the Discriminator
    netD = Discriminator(ngpu, nc, features_disc).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    image_func = lambda im : Image.fromarray(im, 'L')
    if calculate_fid:
        if not os.path.exists(f'fid/run{runid}/reals'):
                        os.makedirs(f'fid/run{runid}/reals')
        for batch_idx, (fid_reals, _) in enumerate(fid_loader):
            if(batch_idx * batch_size == fid_samples):
                break
            reals = fid_reals.reshape(-1, 1, 64, 64).cpu().detach().numpy()
            for i in range(batch_size):
                image_func(reals[i][0].astype('uint8')).save(f"fid/run{runid}/reals/real{i + batch_idx * batch_size:04d}.png")

    # Training Loop

    # Keep track of progress
    iters = 0

    if report_wandb:
        wandb.init(project="mini-gans", entity="taptoi")
        wandb.config = {
                "algorithm": "DCGAN",
                "learning_rate": lr,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "beta1": beta1
                }

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for batch_idx, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            reals = data[0].to(device)
            b_size = reals.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(reals).view(-1)
            # Calculate loss on all-real batch
            errD_reals = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_reals.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fakes = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fakes.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fakes = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fakes.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_reals + errD_fakes
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fakes).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if batch_idx % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, batch_idx, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                if report_wandb:
                    wandb.log({
                        "iteration": iters,
                        "epoch": epoch,
                        "Loss_D": errD.item(),
                        "Loss_G": errG.item()
                    })


            # Check how the generator is doing by saving G's output on fixed_noise
            if batch_idx == 0 and report_wandb:
                with torch.no_grad():
                    fake_samples = netG(fixed_noise).detach().cpu()
                grid_fakes = vutils.make_grid(fake_samples, padding=2, normalize=True)
                grid_reals = vutils.make_grid(reals, padding=2, normalize=True)
                wdb_fakes = wandb.Image(grid_fakes, caption="Fakes")
                wdb_reals = wandb.Image(grid_reals, caption="Reals")
                wandb.log({"fakes": wdb_fakes})
                wandb.log({"reals": wdb_reals})

            if calculate_fid:
                if batch_idx == 0:
                    # generate fakes datasets for FID calculation:
                    if not os.path.exists(f'fid/run{runid}/fakes'):
                        os.makedirs(f'fid/run{runid}/fakes')
                    
                    with torch.no_grad():
                        for batch in range(round(fid_samples / batch_size)):
                            noise_input = torch.randn(64, nz, 1, 1, device=device)
                            fakes = netG(noise_input).reshape(-1, 1, 64, 64)                          
                            for i in range(batch_size):
                                torchvision.utils.save_image(fakes[i][0], f"fid/run{runid}/fakes/fake{i + batch * batch_size:04d}.png",normalize=True)
                    path_fakes = f"fid/run{runid}/fakes"
                    path_reals = f"fid/run{runid}/reals"
                    fid_value = pytorch_fid.fid_score.calculate_fid_given_paths((path_fakes, path_reals), 50, device, fid_samples, 8)
                    if(report_wandb):
                        wandb.log({"FID": fid_value})
                    print("FID value: ", fid_value)

            iters += 1

if __name__ ==  '__main__':
    train()