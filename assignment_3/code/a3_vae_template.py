import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import numpy as np
from scipy.stats import norm

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, input_dim=784, dropout=0):
        super().__init__()

        # initialize layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # self.elu = nn.ELU()

        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_std = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """

        # get mean and logvar
        mean = self.fc_mu(self.dropout(self.relu(self.fc1(input))))
        logvar = self.fc_std(self.dropout(self.relu(self.fc1(input))))
        # std = self.relu(self.fc_std(self.relu(self.dropout(self.fc1(input))))) #may need to add a relu here...#################################

        # std = torch.abs(std) ###########################################

        return mean, logvar


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, output_dim=784, dropout=0):
        super().__init__()

        # initialize layers
        self.fc3 = nn.Linear(z_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # self.elu = nn.ELU()

        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """

        mean = self.sigmoid(self.fc4(self.dropout(self.relu(self.fc3(input)))))

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, image_dim=784, dropout=0, device='cpu'):
        super().__init__()

        # save parameters
        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim, dropout=dropout)
        self.decoder = Decoder(hidden_dim, z_dim, dropout=dropout)
        self.image_dim = image_dim
        self.device = device

    def loss_function(self, input, output, mean, logvar):
        """
        Given input, output, mean and std, compute and return loss
        """

        # to prevent log of 0
        epsilon = 1e-6

        # L_recon = -1*torch.sum(input * torch.log(output + epsilon) + (1-input) * torch.log(1 - output), dim=1) #check if we can use binary CE instead!!!! may need to change mean to sum

        # L_reg = -0.5 * torch.sum(1 + std - mean.pow(2) - std.exp())

        L_recon = torch.nn.functional.binary_cross_entropy(output, input)

        L_reg = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # L_reg = torch.sum(torch.sum(-1*torch.log(std) + ((std.pow(2) + mean.pow(2))-1)*0.5, dim=1, dim=0)

        # Normalise by same number of elements as in reconstruction if we average recon
        L_reg /= input.size(dim=0) * self.image_dim  #####CHECK IF THIS IS NEEDED AGAIN!!!!!!

        # get total loss
        total_loss = torch.mean(L_recon + L_reg,
                                dim=0)  # may need to be the sum###############################################

        return total_loss

    def reparameterize(self, mu, logvar):
        """
        Given the current mu and std, take a random sample epsilon
        from N(0,I), and use these to compute and return z if training,
        otherwise, return mu
        """
        if self.training:

            # convert logvar to std
            std = logvar.mul(0.5).exp_()

            # draw epsilon
            epsilon = torch.randn((1, self.z_dim), device=self.device)

            # compute z
            z = epsilon.mul(std).add_(mu)
            return z

        else:

            return mu

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        # stretch out input into vector
        input = input.view(-1, self.image_dim)

        # get mu and logvar
        mu, logvar = self.encoder(input)

        # get z using the reparameterization trick
        z = self.reparameterize(mu, logvar)

        # generate output
        output = self.decoder(z)

        # get elbo
        average_negative_elbo = self.loss_function(input, output, mu, logvar)

        # free up memory
        input.detach()
        output.detach()
        mu.detach()
        logvar.detach()

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        # sample z
        sampled_z = torch.randn((n_samples, self.z_dim), device=self.device)  # random sample

        # decode z to get mean of images
        im_means = self.decoder(sampled_z)

        # free up memory
        sampled_z.detach()

        # use mean to generate images from bernoullis
        sampled_ims = torch.bernoulli(im_means)

        # reshape vectors into correct dimensions
        sampled_ims = sampled_ims.view(n_samples, 1, 28, 28).to(device=self.device)
        im_means = im_means.view(n_samples, 1, 28, 28).to(device=self.device)

        return sampled_ims, im_means

    def plot_samples(self, n_samples):
        """
        Plot a grid of size n_samples * n_samples with sampled images
        """

        with torch.no_grad():
            sampled_ims, im_means = self.sample(n_samples ** 2)

            # plot all sampled images
            sampled_imgs = make_grid(sampled_ims, nrow=n_samples).numpy().transpose(1, 2, 0)
            plt.figure(1)
            plt.imshow(sampled_imgs)
            plt.show()
            plt.imsave("samples.png", sampled_imgs)

            # plot sampled means
            im_means = make_grid(im_means, nrow=n_samples).numpy().transpose(1, 2, 0)
            plt.figure(2)
            plt.imshow(im_means)
            plt.show()
            plt.imsave("mean_samples.png", im_means)

    def plot_manifold(self, n_samples):
        """
        Plot manifold using a point grid of size n_samples * n_samples
        """

        ppf = norm.ppf(torch.linspace(0.001, 0.999, steps=n_samples))

        # create grid
        x, y = np.meshgrid(ppf, ppf)

        # get z
        grid_z = torch.tensor(np.array(list(zip(x.flatten(), y.flatten())))).to(torch.float).to(device=self.device)

        # decode z to get mean of images
        with torch.no_grad():
            im_means = self.decoder(grid_z)

        im_means = make_grid(im_means.view(-1, 1, 28, 28), nrow=int(np.sqrt(im_means.shape[0] + 2))).numpy().transpose(
            1, 2, 0)
        plt.figure(2)
        plt.imshow(im_means)
        plt.show()
        plt.imsave("manifold.png", im_means)


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average negative elbo for the complete epoch.
    """

    sum_negative_elbo = 0

    for idx, batch in enumerate(data):

        # clear stored gradient
        model.zero_grad()

        # forward pass
        negative_elbo = model(batch)

        # if we are in training mode, backpropagate
        if model.training:
            optimizer.zero_grad()

            # Backward pass
            negative_elbo.backward()
            optimizer.step()

        sum_negative_elbo += negative_elbo.item()

    average_epoch_negative_elbo = sum_negative_elbo / len(data)

    return average_epoch_negative_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average negative elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo', color='darkorange')  # changed colours
    plt.plot(val_curve, label='validation elbo', color='darkblue')
    plt.legend()
    plt.grid(True, lw=0.75, ls='--', c='.75')  # added grid
    plt.xlabel('Epochs')
    plt.ylabel('NEGATIVE ELBO')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def main():
    data = bmnist()[:2]  # ignore test split

    # initialize the device which to run the model on ######################################### ADDED THIS
    device = torch.device(ARGS.device)

    # initialize model and optimizer
    model = VAE(z_dim=ARGS.zdim, dropout=ARGS.dropout)
    optimizer = torch.optim.Adam(model.parameters())

    model.to(device=ARGS.device)

    #set number of samples plotted
    n_samples = 10

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # save model
        # torch.save(model, "VAE_model_epoch_" + str(epoch))

        # plot samples at each epoch
        # model.eval()
        # model.plot_samples(n_samples)
        # model.train()

    # plot manifold at the end of training
    # model.eval()
    # model.plot_manifold(15)
    # model.train()

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device used to train model')  # added this
    parser.add_argument('--dropout', type=float, default=0.2,  # and this
                        help='Dropout probability')

    ARGS = parser.parse_args()

    main()


# For Q 1.14 and Q 1.15

def get_samples_from_epoch(idx, n_samples, z_dim):

    if z_dim == 2:
        path = './results/VAE/2_dim_log_BCE/' # path for a 2-dimensional latent space
    elif z_dim == 20:
        path='./results/VAE/20_dim_log_BCE/' # path for a 20-dimensional latent space
    else:
        print('Error!! Only have z_dim = 2 or z_dim = 20')

    if idx > 99:
        print('Error!! Only have models until epoch 99')
    else:
        # Load the trained model
        model = torch.load(path + 'VAE_model_epoch_' + str(idx), map_location='cpu')
        model.to(ARGS.device)

        model.eval()

        model.plot_samples(n_samples)
        if z_dim == 2:
            model.plot_manifold(n_samples=15)


# plot 10 samples from epoch 99
# get_samples_from_epoch(99, 10, 2)
