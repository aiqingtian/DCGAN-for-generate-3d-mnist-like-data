# Reference: 
# https://github.com/eriklindernoren/Keras-GAN
# https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from datasets import MNIST3D
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument( '--n_epochs',
                     type=int,
                     default=20,
                     help='number of epochs of training' )
parser.add_argument( '--batch_size',
                     type=int,
                     default=64,
                     help='size of the batches' )
parser.add_argument( '--lr',
                     type=float,
                     default=0.0002,
                     help='adam: learning rate' )
parser.add_argument( '--b1',
                     type=float,
                     default=0.5,
                     help='adam: decay of first order momentum of gradient' )
parser.add_argument( '--b2',
                     type=float,
                     default=0.999,
                     help='adam: decay of first order momentum of gradient' )
parser.add_argument( '--n_cpu',
                     type=int,
                     default=8,
                     help='number of cpu threads to use during batch generation' )
parser.add_argument( '--latent_dim',
                     type=int,
                     default=200,
                     help='dimensionality of the latent space' )
parser.add_argument( '--img_size',
                     type=int,
                     default=32,
                     help='size of each image dimension' )
parser.add_argument( '--channels',
                     type=int,
                     default=3,
                     help='number of image channels' )
parser.add_argument( '--sample_interval',
                     type=int,
                     default=400,
                     help='interval between image sampling' )
# These files are already on the VC server. Not sure if students have access to them yet.
parser.add_argument( '--mnist3dtrain_csv',
                     type=str,
                     default='/home/jda93/3DGAN/mnist3d/mnist3d.csv',
                     help='path to the training csv file' )
parser.add_argument( '--mnist3dtrain_root',
                     type=str,
                     default='/home/jda93/3DGAN/mnist3d',
                     help='path to the training root' )

parser.add_argument( '--train_csv',
                     type=str,
                     default='E:/celebA/train.csv',
                     help='path to the training csv file' )
parser.add_argument( '--train_root',
                     type=str,
                     default='E:/celebA/',
                     help='path to the training root' )
opt = parser.parse_args()

class Generator( nn.Module ):
    def __init__( self, d=64 ):
        super( Generator, self ).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(opt.latent_dim, d*8, 2, 2, 0,bias=False),
            nn.BatchNorm3d(d*8),
            nn.ReLU(True),
            nn.ConvTranspose3d(d*8, d*4, 4, 2, 1, bias=False),
            # To generate 64*64*64 fake data, replace above code with below code
            # nn.ConvTranspose3d(d * 8, d * 4, 2, 1, 0, bias=False),
            nn.BatchNorm3d(d*4),
            nn.ReLU(True),
            nn.ConvTranspose3d(d*4, d*2, 4, 2, 1,bias=False),
            # nn.ConvTranspose3d(d * 4, d * 2, 4, 2, 0, bias=False),
            nn.BatchNorm3d(d*2),
            nn.ReLU(True),
            nn.ConvTranspose3d(d*2, d, 4, 2, 1,bias=False),
            nn.BatchNorm3d(d),
            nn.ReLU(True),
            nn.ConvTranspose3d(d, 1, 4, 2, 1,bias=False),
            nn.Sigmoid(),
        )

    # weight_init
    def weight_init( self, mean, std ):
        for m in self._modules:
            normal_init( self._modules[ m ], mean, std )

    # forward method
    def forward( self, x ):
        x = x.view(x.size(0), x.size(1), 1, 1, 1)
        return self.main(x)

class Discriminator(nn.Module):
    # initializers
    def __init__( self, d=64 ):
        super( Discriminator, self ).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(1, d, 4, 2, 1, bias=False),
            nn.BatchNorm3d(d),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(d, 2 * d, 4, 2, 1, bias=False),
            nn.BatchNorm3d(2 * d),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(2 * d, 4 * d, 4, 2, 1, bias=False),
            nn.BatchNorm3d(4 * d),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * d, 8 * d, 4, 2, 1, bias=False),
            nn.BatchNorm3d(8 * d),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8 * d, 1, 2, 2, 0, bias=False),
            nn.Sigmoid()
        )

    # weight_init
    def weight_init( self, mean, std ):
        for m in self._modules:
            normal_init( self._modules[ m ], mean, std )

    # forward method
    def forward( self, x):
        x = self.main(x)
        return x.view(-1, x.size(1))

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')

def voxelize_npy(filename, idx):
    # For displaying generated results saved as .npy files
    imgs = np.load(filename)
    img = imgs[idx]
    img[img>0.5] = 255
    img[img<0.5] = 0
    x, y, z = img.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z', cmap='viridis')
    plt.show()

def main():
    # torch.backends.cudnn.benchmark = True
    cuda = True if torch.cuda.is_available() else False
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
    # Initialize weights
    generator.weight_init( mean=0.0, std=0.02 )
    discriminator.weight_init( mean=0.0, std=0.02 )
    # Configure data loader
    MNIST3D_dataset = MNIST3D()
    dataloader = torch.utils.data.DataLoader(MNIST3D_dataset, batch_size=opt.batch_size,shuffle=True, num_workers=6)
    # Optimizers
    optimizer_G = torch.optim.Adam( generator.parameters(),lr=opt.lr,betas=( opt.b1, opt.b2 ) )
    optimizer_D = torch.optim.Adam( discriminator.parameters(),lr=opt.lr,betas=( opt.b1, opt.b2 ) )
    # ----------
    #  Training
    # ----------
    os.makedirs( 'images', exist_ok=True )
    os.makedirs( 'models', exist_ok=True )
    for epoch in range( opt.n_epochs ):
        # learning rate decay
        if ( epoch + 1 ) == 11:
            optimizer_G.param_groups[ 0 ][ 'lr' ] /= 10
            optimizer_D.param_groups[ 0 ][ 'lr' ] /= 10
            print( 'learning rate change!' )
        if ( epoch + 1 ) == 16:
            optimizer_G.param_groups[ 0 ][ 'lr' ] /= 10
            optimizer_D.param_groups[ 0 ][ 'lr' ] /= 10
            print( 'learning rate change!' )
        for i, (imgs) in enumerate(dataloader):
            # Adversarial ground truths
            real_imgs = imgs.cuda()
            z = torch.randn(imgs.shape[0], opt.latent_dim)
            z = z.cuda()
            gen_imgs = generator(z)
            # Update discrimator
            label_real = discriminator(real_imgs)
            label_gen = discriminator(gen_imgs.detach())
            loss_d = {
                'real_loss': adversarial_loss(label_real, torch.ones_like(label_real)),
                'fake_loss': adversarial_loss(label_gen, torch.zeros_like(label_gen)),
            }
            d_loss = sum(loss_d.values())
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()
            # -----------------
            #  Train Generator
            # -----------------
            d_gen = discriminator(gen_imgs)
            loss_g = {
                'fake_loss': adversarial_loss(d_gen, torch.ones_like(d_gen)),
            }
            g_loss = sum(loss_g.values())
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()
            real_acc = ( label_real > 0.5 ).float().sum() / real_imgs.shape[ 0 ]
            gen_acc = ( label_gen < 0.5 ).float().sum() / gen_imgs.shape[ 0 ]
            d_acc = ( real_acc + gen_acc ) / 2
            print( "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %.2f%%] [G loss: %f]" % \
                    ( epoch,
                      opt.n_epochs,
                      i,
                      len(dataloader),
                      d_loss.item(),
                      d_acc * 100,
                      g_loss.item() ) )
            batches_done = epoch * len( dataloader ) + i
            if batches_done % opt.sample_interval == 0:
                save_img = gen_imgs.clone()
                save_img = torch.squeeze(save_img)
                save_img_numpy = save_img.data.cpu().numpy()
                np.save('/home/jda93/3DGAN/generatedmnist/' + str(batches_done), save_img_numpy)
                torch.save( generator, 'models/gen_%d.pt' % batches_done )
                torch.save( discriminator, 'models/dis_%d.pt' % batches_done )
if __name__ == '__main__':
    main()
