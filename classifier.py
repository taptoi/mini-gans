from __future__ import print_function
#%matplotlib inline
import click
from numpy.random.mtrand import beta
import wandb
import argparse
import os
import glob
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
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from numpy import random


class Classifier(nn.Module):
    def __init__(self, ngpu=1, nc=3, features=64):
        super(Classifier, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Flatten(),
            nn.Linear(features * 8 * 4 * 4, features),
            nn.Linear(features, 10),

        )

    def forward(self, x):
        x = self.main(x)
        x = F.log_softmax(x, dim=1)
        return x


def train(num_epochs):
    runid = random.randint(9999999)
    # Number of workers for dataloader
    workers = 1
    # Batch size during training
    batch_size = 128
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64
    # Number of channels in the training images. For color images this is 3
    nc = 3
    # Size of features in the network
    features = 64
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
    train_dataset = dset.MNIST(root="dataset/", train=True, transform=tforms, download=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
    test_dataset = dset.MNIST(root="dataset/", train=False, transform=tforms, download=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
    fashion_dataset = dset.FashionMNIST(root="dataset/", train=False, transform=tforms, download=True)
    fashion_dataloader = torch.utils.data.DataLoader(fashion_dataset, batch_size=batch_size, 
                                            shuffle=True, num_workers=workers)                                           

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    net = Classifier(ngpu, nc, features).to(device)
    print(net)

    # Run random data through the net as a trivial test
    random_data = torch.rand((1, 3, 64, 64)).to(device)
    result = net(random_data)
    print(result.exp())
    pred, _ = result.exp().max(1)
    print(pred)

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
            for batch_idx, (data,targets) in enumerate(train_dataloader, 0):
                if batch_idx == 0: print(f"training epoch {epoch}")
                data = data.to(device)
                targets = targets.to(device)
                # forward
                output = net(data)
                loss = criterion(output, targets)

                # backward
                optimizer.zero_grad()
                loss.backward()

                # gradient descent on Adam step
                optimizer.step()


    print("Computing accuracy on MNIST training set...")
    compute_accuracy(net, train_dataloader, device)
    print("Computing accuracy on MNIST test set...")
    compute_accuracy(net, test_dataloader, device)
    print("Computing accuracy on Fashion-MNIST set... (should give low accuracy & confidence)")
    compute_accuracy(net, fashion_dataloader, device)

    model_path = f'models/run{runid}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = os.path.join(model_path, f"classifier_net_epoch_{num_epochs}.pt")
    torch.save(net.state_dict(), model_path)
    print(f"Model saved at {model_path}")


def compute_accuracy(model, dataloader, device):

    ncorrect = 0
    nsamples = 0
    avg_confidence = 0.0

    # switch the net into inference mode
    model.eval()

    with torch.no_grad():
        for xs, ys in dataloader:
            xs = xs.to(device)
            ys = ys.to(device)

            output = model(xs)
            _, preds = output.max(1)
            ncorrect += (preds == ys).sum()
            # get confidence levels of the predictions
            # get probabilities from log softmax
            probs, _ = output.exp().max(1)
            avg_confidence += probs.sum()
            nsamples += preds.size(0)

    avg_confidence = avg_confidence / float(nsamples) * 100.
    accuracy = float(ncorrect) / float(nsamples) * 100.
    print(f'Correct classifications: {ncorrect}/{nsamples}')
    print(f'Accuracy: {accuracy:2f}')
    print(f'Average confidence: {avg_confidence:2f}')

def infer(in_folder, model_path):
    net = Classifier()
    net.load_state_dict(torch.load(model_path))
    image_list = []
    image_classes = []
    
    for filename in glob.glob(f'{in_folder}/*.png'):
        im=Image.open(filename)
        image_list.append(im)
    for im in image_list:
        convert_tensor = transforms.ToTensor()
        tensor = convert_tensor(im).reshape((1, 3, 64, 64))
        _,output = net(tensor).max(1)
        image_classes.append(output.numpy()[0])
    return image_classes

@click.command()
@click.option('--entry_point', default="train", help='Entry point function. train | infer')
@click.option('--num_epochs', default=5, help='Number of epochs to train.')
@click.option('--in_folder', default="", help='Path to the folder containing the images to classify')
@click.option('--model_path', default="", help='Path to the pretrained model')
def entry(entry_point, num_epochs, in_folder, model_path):
    if(entry_point == "train"):
        train(num_epochs)
    if(entry_point == "infer"):
        infer(in_folder, model_path)
    else:
        return

if __name__ ==  '__main__':
    entry()