import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from datasets.mnist import mnist
import os
from torchvision.utils import make_grid
import numpy as np


def log_prior(x):
    """
    Compute the elementwise log probability of a standard Gaussian, i.e.
    N(x | mu=0, sigma=1).
    """

    logp = (-0.5 * x.pow(2) - 0.5*torch.log(2 * torch.tensor(np.pi))).sum(dim=1)

    return logp


def sample_prior(size):
    """
    Sample from a standard Gaussian.
    """

    sample = torch.randn(size, device=ARGS.device)

    return sample


def get_mask():
    mask = np.zeros((28, 28), dtype='float32')
    for i in range(28):
        for j in range(28):
            if (i + j) % 2 == 0:
                mask[i, j] = 1

    mask = mask.reshape(1, 28 * 28)
    mask = torch.from_numpy(mask)

    return mask


class Coupling(torch.nn.Module):
    def __init__(self, c_in, mask, n_hidden=1024, device='cpu'):
        super().__init__()
        self.n_hidden = n_hidden
        self.device=device

        # Assigns mask to self.mask and creates reference for pytorch.
        self.register_buffer('mask', mask)

        # Create shared architecture to generate both the translation and
        # scale variables.
        # Suggestion: Linear ReLU Linear ReLU Linear.
        self.shared_net = torch.nn.Sequential(nn.Linear(c_in, n_hidden),
                                              nn.ReLU(),
                                              nn.Linear(n_hidden, n_hidden),
                                              nn.ReLU(),
                                              )

        # initialize last layers for scale and translation nets
        self.scale_net = nn.Linear(n_hidden, c_in)
        self.translate_net = nn.Linear(n_hidden, c_in)

        # initialize tanh layer for scale_net
        self.tanh = nn.Tanh()

        # The nn should be initialized such that the weights of the last layer
        # is zero, so that its initial transform is identity.

        self.scale_net.weight.data.zero_()
        self.translate_net.weight.data.zero_()
        self.scale_net.bias.data.zero_()
        self.translate_net.bias.data.zero_()

    def forward(self, z, ldj, reverse=False):
        # Implement the forward and inverse for an affine coupling layer. Split
        # the input using the mask in self.mask. Transform one part with
        # Make sure to account for the log Jacobian determinant (ldj).
        # For reference, check: Density estimation using RealNVP.

        # NOTE: For stability, it is advised to model the scale via:
        # log_scale = tanh(h), where h is the scale-output
        # from the NN.

        mask = self.mask

        # get masked z
        masked_z = z * mask

        # get scale and translation
        log_s = self.tanh(self.scale_net(self.shared_net(masked_z)))
        t = self.translate_net(self.shared_net(masked_z))

        if not reverse:

            # transform z using the affine coupling layer
            z = masked_z + (1 - mask) * (z * torch.exp(log_s) + t)

            # compute determinant of transformation
            ldj += ((1 - self.mask) * log_s).sum(dim=1)


        else:

            # comptute inverse transformation of z
            z = masked_z + (1 - mask) * ((z - t) * torch.exp(-log_s))

            # set determinant to 0
            ldj = torch.zeros(ldj.shape)

        return z, ldj


class Flow(nn.Module):
    def __init__(self, shape, n_flows=4, device='cpu'):
        super().__init__()
        channels, = shape

        mask = get_mask()

        self.layers = torch.nn.ModuleList()

        for i in range(n_flows):
            self.layers.append(Coupling(c_in=channels, mask=mask, device=device))
            self.layers.append(Coupling(c_in=channels, mask=1 - mask, device=device))

        self.z_shape = (channels,)


    def forward(self, z, logdet, reverse=False):
        if not reverse:
            for layer in self.layers:
                z, logdet = layer(z, logdet)
        else:
            for layer in reversed(self.layers):
                z, logdet = layer(z, logdet, reverse=True)

        return z, logdet


class Model(nn.Module):
    def __init__(self, shape, device='cpu'):
        super().__init__()
        self.flow = Flow(shape, device=device)
        self.device = device

    def dequantize(self, z):
        return z + torch.rand_like(z)

    def logit_normalize(self, z, logdet, reverse=False):
        """
        Inverse sigmoid normalization.
        """
        alpha = 1e-5

        if not reverse:
            # Divide by 256 and update ldj.
            z = z / 256.
            logdet -= np.log(256) * np.prod(z.size()[1:])

            # Logit normalize
            z = z * (1 - alpha) + alpha * 0.5
            logdet += torch.sum(-torch.log(z) - torch.log(1 - z), dim=1)
            z = torch.log(z) - torch.log(1 - z)

        else:
            # Inverse normalize
            logdet += torch.sum(torch.log(z) + torch.log(1 - z), dim=1)
            z = torch.sigmoid(z)

            # Multiply by 256.
            z = z * 256.
            logdet += np.log(256) * np.prod(z.size()[1:])

        return z, logdet

    def forward(self, input):
        """
        Given input, encode the input to z space. Also keep track of ldj.
        """
        z = input
        ldj = torch.zeros(z.size(0), device=z.device)

        z = self.dequantize(z)
        z, ldj = self.logit_normalize(z, ldj)

        z, ldj = self.flow(z, ldj)

        # get log probability distribution over z
        log_pz = log_prior(z)

        # transform log probability distribution to that of x
        log_px = log_pz + ldj

        return log_px

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Sample from prior and create ldj.
        Then invert the flow and invert the logit_normalize.
        """
        z = sample_prior((n_samples,) + self.flow.z_shape, device=self.device)
        ldj = torch.zeros(z.size(0), device=z.device)

        # invert the flow and logit normalize
        z, ldj = self.flow(z, ldj, reverse=True)
        z, _ = self.logit_normalize(z, ldj, reverse=True)

        return z.reshape(n_samples,1,28,28).long().to(device=ARGS.device)

    def plot_samples(self, n_samples):
        """
        Plot a grid of size n_samples * n_samples with sampled images
        """
        with torch.no_grad():

            gen_imgs = self.sample(n_samples ** 2)

            # plot all sampled images
            gen_imgs = make_grid(gen_imgs, nrow=n_samples).numpy().transpose(1, 2, 0)
            plt.figure(1)
            plt.imshow(gen_imgs)
            plt.show()
            plt.imsave("gen.png", gen_imgs)


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average bpd ("bits per dimension" which is the negative
    log_2 likelihood per dimension) averaged over the complete epoch.
    """

    losses = []

    for idx, (batch, _) in enumerate(data):

        # send data to device
        batch.to(ARGS.device)

        # clear stored gradient
        model.zero_grad()

        # forward pass
        log_px = model.forward(batch)


        # free up memory
        batch.detach()

        # get loss
        loss = -log_px.mean(dim=0)

        if model.training:
            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # clip gradient
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=10)

            optimizer.step()

        losses.append(loss.item())

        # if idx % 200 == 0:
        #     bpd = loss/(28 * 28 * np.log(2))
        #     print(f"[Batch {idx}] bpd: {bpd.item()} ")
        #     model.eval()
        #     model.plot_samples(5)
        #     model.train()

    # get average epoch loss
    epoch_loss = np.mean(losses)

    # get average bpd
    avg_bpd = epoch_loss / (28 * 28 * np.log(2))

    return avg_bpd


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average bpd for each.
    """
    traindata, valdata = data

    model.train()
    train_bpd = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_bpd = epoch_iter(model, valdata, optimizer)

    return train_bpd, val_bpd


def save_bpd_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train bpd', color='darkorange')
    plt.plot(val_curve, label='validation bpd', color='darkblue')
    plt.legend()
    plt.grid(True, lw=0.75, ls='--', c='.75')  # added grid
    plt.xlabel('epochs')
    plt.ylabel('bpd')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = mnist()[:2]  # ignore test split

    model = Model(shape=[784], device=ARGS.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs('images_nfs', exist_ok=True)

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        bpds = run_epoch(model, data, optimizer)
        train_bpd, val_bpd = bpds
        train_curve.append(train_bpd)
        val_curve.append(val_bpd)
        print("[Epoch {epoch}] train bpd: {train_bpd} val_bpd: {val_bpd}".format(
            epoch=epoch, train_bpd=train_bpd, val_bpd=val_bpd))

        model.eval()
        model.plot_samples(5)
        model.train()

        # save model
        if epoch % 50 == 0:
            torch.save(model, "NF_epoch_" + str(epoch))

    save_bpd_plot(train_curve, val_curve, 'nfs_bpd.pdf')

    # save final model
    torch.save(model, "NF_final")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device used to train model')
    ARGS = parser.parse_args()

    main()
