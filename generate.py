import os
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def generate(config):
  model = torch.load(os.path.join(config.weights, 'mnist_vae.pt'))
  latent_size = 15
  z = torch.randn(10, latent_size).to(device='cuda')
  model.eval()
  samples = model.decoder(z).data.cpu().numpy()

  plt.figure(figsize=(10, 1))
  gspec = gridspec.GridSpec(1, 10)
  gspec.update(wspace=0.05, hspace=0.05)
  for i, sample in enumerate(samples):
    ax = plt.subplot(gspec[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    plt.savefig(os.path.join(config.output, 'vae_generation.jpg'))

if __name__ == '__main__':
  args = argparse.ArgumentParser(description='Generate MNIST_VAE',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  args.add_argument('-weights', type=str, default='./',
    help='Path to weight file. Ex: "./"')
  args.add_argument('-output', type=str, default='./',
    help='Path to save generated image. Ex: "./model"')

  config = args.parse_args()
  generate(config)
   