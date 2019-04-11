# Perform a linear interpolation between two latent space
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

z1 = torch.randn(64, 200)
z2 = torch.randn(64, 200)
generator = torch.load('models/gen_6400.pt')
generator = generator.cuda()
generator.eval()
z1 = z1.cuda()
z2 = z2.cuda()
os.makedirs('img', exist_ok=True)

for i in range(10):
    n1 = 0.1*i
    n2 = 1 - n1
    z = n1*z1 + n2*z2
    gen_imgs = generator(z)
    sample_img = gen_imgs.data[0]
    sample_img = sample_img.view(32, 32, 32)
    sample_img = sample_img.cpu().numpy()
    sample_img[sample_img>0.5] =255
    sample_img[sample_img<0.5] = 0
    sample_img = np.transpose(sample_img, (2, 1, 0))
    x, y, z = sample_img.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z', cmap='viridis')
    fig.savefig('img/%d.png' % i)
    plt.close(fig)