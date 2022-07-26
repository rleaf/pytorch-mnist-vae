import os
from model.model import VAE
from model.loss import loss as loss_function
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as T
import argparse


def one_hot(labels, class_size):
   targets = torch.zeros(labels.size(0), class_size)
   for i, label in enumerate(labels):
      targets[i, label] = 1
   return targets

def train_vae(epoch, model, train_loader, cond=False):
   model.train()
   train_loss = 0
   num_classes = 10
   optimizer = optim.Adam(model.parameters(), lr=1e-3)
   for batch_idx, (data, labels) in enumerate(train_loader):
      data = data.to(device=next(model.parameters()).device)
      # data = data.to(device='cuda')
      if cond:
         one_hot_vec = one_hot(labels, num_classes).to(device='cuda')
         recon_batch, mu, logvar = model(data, one_hot_vec)
      else:
         recon_batch, mu, logvar = model(data)
      optimizer.zero_grad()
      loss = loss_function(recon_batch, data, mu, logvar)
      loss.backward()
      train_loss += loss.data
      optimizer.step()
   print('Train Epoch: {} \tLoss: {:.6f}'.format(
      epoch, loss.data))


def main(config):
   # print(config)
   num_epochs = config.epoch
   latent_size = config.latent_size
   batch_size = config.batch_size
   input_size = config.input_size
   device = config.cuda
   # return
   vae_model = VAE(input_size, latent_size=latent_size)


   mnist_train = dset.MNIST('{}MNIST_data'.format(config.train_path), train=True, download=True,
                           transform=T.ToTensor())
   loader_train = DataLoader(mnist_train, batch_size=batch_size,
                           shuffle=True, drop_last=True, num_workers=2)
   
   if device and torch.cuda.is_available():
      vae_model.cuda()
      # mnist_train.train_data.to(device=next(vae_model.parameters()).device)
   else:
      vae_model.cpu()

   for epoch in range(0, num_epochs):
      train_vae(epoch, vae_model, loader_train)
   
   if config.save_weights:
      torch.save(vae_model, os.path.join('./', 'mnist_vae.pt'))

if __name__ == '__main__':
   args = argparse.ArgumentParser(description='MNIST_VAE',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

   args.add_argument('-d_z', '--latent_size', type=int, default=15)
   args.add_argument('-b', '--batch_size', type=int, default=128)
   args.add_argument('-i', '--input_size', type=int, default=784) # 28*28
   args.add_argument('-train_path', type=str, default='./',
      help='path/to/training/data')
   args.add_argument('-epoch', type=int, default=10)
   args.add_argument('-c', '--cuda', action='store_false',
      help='use GPU if available, otherwise use CPU')
   args.add_argument('-s', '--save_weights', action='store_false',
      help='save weights to local directory')

   config = args.parse_args()

   main(config)
