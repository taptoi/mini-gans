import click
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image 
import PIL 
import os
from numpy import random
import numpy as np
import pytorch_fid.fid_score
from torch.utils.tensorboard import SummaryWriter

# Discriminator judges the image if it is a fake or real
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,1), # output single value (fake 0, real 1)
            nn.Sigmoid(), # ensure 0 or 1 at output
        )

    def forward(self, x):
        return self.disc(x)

# Generator creates the fakes
# generator uses a prior distribution of noise to 
# produce a wide range of diverse samples. 
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim): # z_ dim is noise dimension
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim), # 28x28x1 -> 784
            nn.Tanh(), # ensure output of pixel vas is between -1 and 1
        )

    def forward(self, x):
        return self.gen(x)


@click.command()
@click.option('--num_epochs', default=50, help='Number of epochs to train.')
@click.option('--report_tensorboard', default=False, help='Use tensorboard for reporting.')
@click.option('--report_wandb', default=False, help='Use weights & biases for reporting.')
@click.option('--calculate_fid', default=False, help='Calculate the Frechet inception distance metric between fakes and reals.')
def train(num_epochs, report_tensorboard, report_wandb, calculate_fid):
    runid = random.randint(9999999)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate = 3e-4
    z_dim = 64
    image_dim = 28 * 28 * 1
    batch_size = 32

    fid_samples = 2048

    disc = Discriminator(image_dim).to(device)
    gen = Generator(z_dim, image_dim).to(device)
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)
    tforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5))]
    )
    dataset = datasets.MNIST(root="dataset/", transform=tforms, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    fid_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt_disc = optim.Adam(disc.parameters(), lr=learning_rate)
    opt_gen = optim.Adam(gen.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
    writer_real = SummaryWriter(f"runs/GAN_MNIST/real")

    if report_wandb:
        wandb.init(project="mini-gans", entity="taptoi")
        wandb.config = {
                "learning_rate": learning_rate,
                "epochs": num_epochs,
                "batch_size": batch_size
                }

    image_func = lambda im : Image.fromarray(im, 'L')

    if calculate_fid:
        if not os.path.exists(f'fid/run{runid}/reals'):
                        os.makedirs(f'fid/run{runid}/reals')
        for batch_idx, (fid_reals, _) in enumerate(fid_loader):
            if(batch_idx * batch_size == fid_samples):
                break
            reals = fid_reals.reshape(-1, 1, 28, 28).cpu().detach().numpy()
            for i in range(batch_size):
                image_func(reals[i][0].astype('uint8')).save(f"fid/run{runid}/reals/real{i + batch_idx * batch_size:04d}.png")

    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]

            ### Train Discruminator: max log(D(real)) 0 log(1 - D(G(z))
            # sample minibatch of noise samples from noise prior p_g(z)
            noise = torch.randn(batch_size, z_dim).to(device)
            # sample minibatch of examples from data generating distribution
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = 0.5 * (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward(retain_graph=True) # to preserve fake computation
            opt_disc.step()

            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()
                        
            if calculate_fid:
                if batch_idx == 0:
                    # generate fakes datasets for FID calculation:
                    if not os.path.exists(f'fid/run{runid}/fakes'):
                        os.makedirs(f'fid/run{runid}/fakes')
                    
                    with torch.no_grad():
                        for batch in range(round(fid_samples / batch_size)):
                            noise_input = torch.randn(batch_size, z_dim).to(device)
                            fakes = gen(noise_input).reshape(-1, 1, 28, 28)                          
                            for i in range(batch_size):
                                torchvision.utils.save_image(fakes[i][0], f"fid/run{runid}/fakes/fake{i + batch * batch_size:04d}.png",normalize=True)
                    path_fakes = f"fid/run{runid}/fakes"
                    path_reals = f"fid/run{runid}/reals"
                    fid_value = pytorch_fid.fid_score.calculate_fid_given_paths((path_fakes, path_reals), 50, device, fid_samples, 8)
                    if(report_wandb):
                        wandb.log({"FID": fid_value})
                    print("FID value: ", fid_value)


            if report_tensorboard:
                if batch_idx == 0:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}] \ "
                        f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
                    )

                    with torch.no_grad():
                        fakes = gen(fixed_noise).reshape(-1, 1, 28, 28)
                        reals = real.reshape(-1, 1, 28, 28)
                        img_grid_fake = torchvision.utils.make_grid(fakes, normalize=True)
                        img_grid_real = torchvision.utils.make_grid(reals, normalize=True)

                        writer_fake.add_image(
                            "Mnist fakes", img_grid_fake, global_step=epoch
                        )
                        writer_real.add_image(
                            "Mnist reals", img_grid_real, global_step=epoch
                        )
                        
            
            if report_wandb:
                if batch_idx == 0:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}] \ "
                        f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
                    )
                    wandb.log({"lossD": lossD, 
                                "lossG": lossG,
                                "epoch": epoch,
                                })
                    fakes = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    reals = real.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fakes, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(reals, normalize=True)
                    wdb_fakes = wandb.Image(img_grid_fake, caption="Fakes")
                    wdb_reals = wandb.Image(img_grid_real, caption="Reals")
                    wandb.log({"fakes": wdb_fakes})
                    wandb.log({"reals": wdb_reals})

if __name__ == '__main__':
    train()