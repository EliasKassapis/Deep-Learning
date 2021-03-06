import argparse
import os

import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from torchvision import datasets

# set default tensor types
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

        # initialize generator network

        # recommended
        self.generator = nn.Sequential(nn.Linear(args.latent_dim, 128),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.2),
                                       nn.Linear(128, 256),
                                       nn.BatchNorm1d(256),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.2),
                                       nn.Linear(256, 512),
                                       nn.BatchNorm1d(512),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.2),
                                       nn.Linear(512, 1024),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.2),
                                       nn.Linear(1024, 784),
                                       nn.Tanh())

    def forward(self, z):
        # Generate images from z

        output = self.generator(z)

        return output

    def generate_samples(self, n_samples):
        """
        Generate n_samples from the Generator.
        """

        # sample z
        sampled_z = torch.randn(n_samples, args.latent_dim, device=args.device)  # random sample

        gen_imgs = self.forward(sampled_z)

        # reshape images into correct dimensions
        gen_imgs = gen_imgs.view(-1, 1, 28, 28).to(device=args.device)
        # im_means = im_means.view(n_samples, 1, 28, 28).to(device=self.device)

        return gen_imgs

    def plot_samples(self, n_samples):
        """
        Plot a grid of size n_samples * n_samples with sampled images
        """

        with torch.no_grad():
            gen_imgs = self.generate_samples(n_samples ** 2)

            # plot all sampled images
            gen_imgs = make_grid(gen_imgs, nrow=n_samples).cpu().numpy().transpose(1, 2, 0)
            plt.figure(1)
            plt.imshow(gen_imgs)
            plt.show()
            plt.imsave("gen.png", gen_imgs)


    def plot_interpolation(self, n_samples, start_value, end_value): ###############################################################################################
        #     """
        #     Plot interpolation n_samples * n_samples
        #     """

        # get dimension values for each z
        dim_values = torch.linspace(start_value, end_value, n_samples).to(device=args.device)

        # initialize z
        z_ones = torch.ones((n_samples, args.latent_dim), device=args.device)

        # get batch of interpolated z's
        intp_z = dim_values * z_ones.t()

        # generate images
        intp_imgs = self.forward(intp_z.t())

        # reshape images into correct dimensions
        intp_imgs = intp_imgs.view(-1, 1, 28, 28).to(device=args.device)


        # int_imgs = make_grid(int_imgs.view(-1, 1, 28, 28), nrow=int(np.sqrt(int_imgs.shape[0] + 2))).detach().numpy().transpose(1, 2, 0)
        intp_imgs = make_grid(intp_imgs.view(-1, 1, 28, 28), nrow=9).detach().cpu().numpy().transpose(1, 2, 0)

        plt.figure(2)
        plt.imshow(intp_imgs)
        plt.show()
        plt.imsave("int_imgs.png", intp_imgs)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

        # initialize discriminator network

        self.discriminator = nn.Sequential(nn.Linear(784, 512),
                                           nn.LeakyReLU(0.2),
                                           nn.Dropout(0.2),
                                           nn.Linear(512, 256),
                                           nn.BatchNorm1d(256),
                                           nn.LeakyReLU(0.2),
                                           nn.Dropout(0.2),
                                           nn.Linear(256, 1),
                                           nn.Sigmoid()
                                           )

    def forward(self, img):
        # return discriminator score for img

        score = self.discriminator(img)

        return score


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    G_losses = []
    D_losses = []

    G_avg_epoch_loss = []
    D_avg_epoch_loss = []

    for epoch in range(args.n_epochs):

        G_epoch_loss = []
        D_epoch_loss = []

        for i, (imgs, _) in enumerate(dataloader):

            batch_size = imgs.shape[0]

            # strech out images into vectors
            real_imgs = imgs.view(-1, 784).to(args.device)

            # Train Generator
            # ---------------

            # clear stored gradient
            generator.zero_grad()

            # forward pass
            z = torch.randn(batch_size, args.latent_dim, device=args.device)  # sample z
            gen_imgs = generator(z)  # generate images

            gen_score = discriminator(gen_imgs)  # get score

            # free up memory
            gen_imgs.detach()

            # get expected loss over batch using non-saturating heuristic
            G_loss = -1 * torch.mean(torch.log(gen_score))
            G_losses.append(G_loss.item())
            G_epoch_loss.append(G_loss.item())
            optimizer_G.zero_grad()

            # backward pass
            G_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------

            # clear stored gradient
            discriminator.zero_grad()

            # forward pass
            real_score = discriminator(real_imgs)
            z = torch.randn(batch_size, args.latent_dim, device=args.device)  # sample z
            gen_score = discriminator(generator(z))

            # free up memory
            real_imgs.detach()
            z.detach()

            # to prevent math error
            epsilon = 1e-6

            # get expected loss over batch
            real_loss = -1 * torch.mean(torch.log(real_score))
            gen_loss = -1 * torch.mean(torch.log(1 - gen_score))

            D_loss = real_loss + gen_loss

            # D_loss = -1 * torch.mean(torch.log(real_score + epsilon) + torch.log(1 - gen_score))
            D_losses.append(D_loss.item())
            D_epoch_loss.append(D_loss.item())
            optimizer_D.zero_grad()

            # backward pass
            D_loss.backward()
            optimizer_D.step()

            if i % 200 == 0:
                print(f"[Epoch {epoch}, Batch {i}] D loss: {D_loss.item()} G loss: {G_loss.item()}")

                # # generate sample images
                # generator.eval()
                # generator.plot_samples(10)
                # generator.train()


        G_avg_epoch_loss.append(np.mean(G_epoch_loss))
        D_avg_epoch_loss.append(np.mean(D_epoch_loss))


        # save model
        if epoch % 50 == 0:
            torch.save(generator, "GAN_generator_epoch_" + str(epoch))
            torch.save(discriminator, "GAN_discriminator_epoch_" + str(epoch))


    return G_avg_epoch_loss, D_avg_epoch_loss


def save_loss_plot(g_curve, d_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(g_curve, label='G Loss', color='darkorange')  # changed colours
    plt.plot(d_curve, label='D Loss', color='darkblue')
    plt.legend()
    plt.grid(True, lw=0.75, ls='--', c='.75')  # added grid
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.55,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models
    generator = Generator()
    discriminator = Discriminator()

    generator.to(device=args.device)
    discriminator.to(device=args.device)

    # Initialize optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Start training
    g_loss, d_loss = train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # save g_loss and d_loss
    np.save("g_loss", g_loss)
    np.save("d_loss", d_loss)

    # save model
    torch.save(generator, "GAN_generator_final")
    torch.save(discriminator, "GAN_discriminator_final")

    # save losses
    save_loss_plot(g_loss, d_loss, 'GAN_loss.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--dropout_prob', type=float, default=0.2,
                        help='Dropout probability')
    parser.add_argument("--b1", type=float, default=0.5,  # from InfoGAN
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device used to train model')

    # parser.add_argument('--device', type=str, default=('cpu'),
    #                     help='Device used to train model')


    args = parser.parse_args()

    main()


# For Q 2.6

def get_samples_from_epoch(idx, n_samples):
    path = './results/GAN/run 3/'
    path = ''

    # Load the trained model
    # generator = torch.load(path + 'GAN_generator_epoch_' + str(idx), map_location='cpu')
    generator = torch.load(path + 'GAN_generator_final', map_location='cpu')

    # discriminator = torch.load(path + 'GAN_discriminator_epoch_' + str(idx), map_location='cpu')
    generator.to(args.device)

    generator.eval()
    generator.plot_samples(n_samples)
    # generator.plot_interpolation(9, -0.75, -0.5)
    generator.train()




# plot 10 samples from epoch 99
# get_samples_from_epoch(100, 5)



